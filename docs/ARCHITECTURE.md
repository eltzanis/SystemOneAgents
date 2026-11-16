# How a System 1 agent works

A System 1 agent calls MCP tools by answering typed multiple-choice questions; it never generates text. This guide
explains the idea, the three components, how they connect, and the method we used to develop them. Code paths refer
to `code/src/`.

## 1. The idea

A generating language model that acts as an agent writes its tool call token by token, so every tool name, path and
number it emits can be misspelled, paraphrased or invented. A **decision model** reads a state, a question and a
finite list of options, and returns a probability for every option in one forward pass, without sampling.

To build an agent on a decision model, every step that an LLM agent performs by writing must become a choice among
written options. Three facts make that possible for tool calling:

- The tools to choose from are listed, with their schemas, in the MCP catalogue.
- Every argument value a correct call needs comes from one of four places:
  - it is written in the request (a path, a number, a name);
  - it is derived from what is written by a fixed rule (a unit conversion);
  - it is listed in the tool's schema (an enumeration, a documented value);
  - it is set by the agent's own policy (an output folder).
- Whether to call a tool, decline, ask the user or stop is itself a closed decision.

A value that is none of these cannot be produced; the agent asks the user for it instead.

This gives three properties by construction:

- a call contains only values taken from the request, the schema or the policy (no fabricated content; verified on
  every call of the study);
- every decision carries a probability that thresholds can act on;
- the same input always gives the same output, so every run can be replayed.

## 2. Components

| component | code | built as | inputs | outputs | responsible for |
|---|---|---|---|---|---|
| Agent | `s1agent/agent.py` | deterministic code | the request, the MCP catalogue, earlier steps | a tool call with its confidence, a decline, or a question to the user | the order of steps, thresholds, policy defaults, asking, calling the MCP server |
| Compiler | `s1agent/compiler.py`, `s1agent/candidates.py` | deterministic code | the request, the chosen tool's JSON schema, earlier steps | argument values; required values missing or uncertain | which questions fill each argument, their options, how answers combine |
| Decision model | `s1agent/backends/` | a published model, unchanged, served by llama.cpp | one question: state, question, options, kind | a probability for every option | every judgement of meaning |

A question is a state string, a question string, an option list and a kind (a choice, or a yes/no question); its answer
is one probability per option (`backends/base.py`, `Decision`). Any model that answers this interface can be used
without changing the agent or the compiler.

**How the components depend on each other:**

- The **decision model** depends on the agent and the compiler for everything it can say. It only picks among the
  options it is shown, so a value that no question offers can never appear in a call.
- The **compiler** depends on the agent for the tool to fill, and on the model for every judgement. It never decides
  what a request means.
- The **agent** depends on both, and adds the policy.

## 3. The agent (`agent.py`)

The state string is `User request: <request>`, followed by any earlier steps.

1. **Tool selection.** One question: "Which tool should be called next for this request?". The options are every
   tool (`server / name: the first 220 characters of its description`) plus "none of these: the request is not
   something these tools can do". The last option declines.
2. **Tool fit.** When the two most probable tools are within 0.3 of each other, a second question shows those two
   tools with their parameter names and asks which one has parameters for the values the request gives.
3. **Capability check.** A yes/no question, "Can the tool `x` do what the request asks?", with the whole description
   and the parameter names. Below P(yes) = 0.5 the agent declines.
4. **Argument filling** by the compiler.
5. **Ask or call.** The agent declines below a selection confidence of 0.3. It asks the user for missing or uncertain
   required values. Otherwise it calls the tool over MCP.

For several calls, each call and the first 1,200 characters of its result are added to the state and become candidate
values. A question ("Is the request fully handled, or is another tool call needed?") decides whether to continue.

Settings are in `configs/agent.yaml`.

## 4. The compiler (`compiler.py`, `candidates.py`)

**Argument kinds.** Each property gets a kind from its type, name and description:

- enumeration, boolean, number;
- path (including string arguments named or described as files);
- column, free text;
- list, array of arrays, object.

A union type (anyOf/oneOf, such as pydantic's Optional[X]) is read as its single non-null type.

**Questions:**

- **Enumeration:** "Which value should the argument `x` take? `x`: <description>". The options are the listed values,
  plus "not specified in the request (keep the default)" for an optional argument.
- **Boolean:** "Does the request turn the option `x` on or off?". The options are "yes: turn it on", "no: turn it off"
  and, for an optional argument, "not mentioned in the request (keep the default)".
- **Number, path, column, free text:** the same value question, over the candidate values plus "not specified".
- **List:** items are picked one at a time. "Which item does the request want in `x`?" is asked over the remaining
  candidates plus "none: the request names no item for it", then asked again with "Already chosen: …". This way
  candidates compete with each other.
- **Arrays of arrays and objects:** written lists and dictionaries are offered whole. Otherwise an object is filled by
  a yes/no gate, then its keys (picked like list items), then one value question per key.

**Candidate values:**

- **The request:**
  - paths, including quoted paths with spaces;
  - numbers in digits or words, in scientific notation, with thousands separators, or as ranges;
  - quoted strings, recognised strictly;
  - identifiers and compact codes;
  - words and phrases of up to three words, for name and search arguments;
  - written lists and dictionaries;
  - text after a label.
- **Derived forms** (only for arguments that take that kind of value):
  - unit conversions and fractions;
  - class counts;
  - dataset identifiers;
  - chromosome and allele notations;
  - paths composed from written parts.
- **The schema:** listed values, with closed sets offered in full, and documented list items.
- **The environment:** the header of a CSV file another argument points to, and results of earlier calls.

Single words that name the tool or its server are not offered.

**Settling the answers:**

- Independent questions are sent together, then the answers settle in a fixed order.
- Free values settle highest probability first.
- A value fills at most as many arguments as the request writes it.
- Used values stay visible as options but cannot be picked again.
- When two arguments claim the same value, the model is asked which argument it belongs to; that answer decides only
  if it reaches 0.7.

**Acceptance rules:**

- An optional value is set only when the probability on its allowed values reaches 0.7, and a list's first item must
  reach 0.7 too.
- A value equal to the documented default is not passed.
- A required argument without a value is reported as missing, and one below 0.5 as uncertain.

## 5. The decision models (`backends/`)

| family | models | how a question is asked and read |
|---|---|---|
| decider | decider-4b, decider-35b-a3b | `Context: … Question: … Options: (A) … (B) … Answer: (` as token ids. The label tokens' log-probabilities are divided by the fitted temperature and normalised over the valid labels. Labels A–J, or one single-token label per option up to 255. |
| pplx-decider | pplx-decider-flash, pplx-decider-27b | The chat template around a fixed system prompt and `State: … Question 0: … Options: A: …`, ending in a marker. llama-server returns the marker's final hidden state (embeddings mode, non-causal attention). The read-out matrix and temperature are applied in `pplx_llamacpp.py`. |
| native SystemOne | OpenJev, Clef | llama-server's `/v1/systemone` takes the state and one question whose criteria are the option texts. OpenJev takes at most 52 options; larger questions are answered in balanced groups, then among the group winners. |

Models are deterministic: the same question gives the same distribution. Every answer is stored under the SHA-256 of
(model, kind, state, question, options) (`AnswerStore`), and replaying the stored answers reproduces a run exactly.

## 6. A method to develop System 1 agents

1. **Separate judgement from mechanics.** Every text the call needs reaches the model as an option. Anything code can
   do exactly is done by the compiler.
2. **Measure the compiler with an oracle** (`scripts/oracle.py`). A backend that knows the label picks the labelled
   value whenever it is offered. The requests it cannot complete show what the compiler cannot express: an upper
   bound for any model, measured without a GPU.
3. **Classify failures.** A failure is either a compiler failure (the value was never offered, or offered in the
   wrong form), a model failure (offered but not chosen) or a label error. Only compiler failures are fixed in code.
4. **Change the compiler by general rules, declared before measuring.** A rule covers a way requests write values or
   schemas describe arguments, never a particular tool.
5. **Accept by fixed criteria.** The summed number of fully correct calls over the models must rise on development
   requests, earlier reference requests must not be lost beyond a tolerance, and a leave-one-out run must not show
   that removing a change helps.
6. **Store answers.** A changed compiler asks the models only its new questions, and every result of a run can be
   recomputed from its stored answers without a GPU.
7. **Freeze, then test once,** on requests written separately and audited.
