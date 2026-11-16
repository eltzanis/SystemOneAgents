"""Decision models that llama-server answers natively through POST /v1/systemone (llama.cpp b11929 and later).

llama.cpp implements several decision models end to end, including their prompt, readout head and stored temperatures
(among them Clef and OpenJev). The server takes a request of named questions and returns one probability per option, so
this backend only maps one compiler question to that request:

    {"state": <state>, "questions": {"q": {"type": "choice", "instructions": <question>,
                                           "criteria": {<option>: null, <option>: null, ...}}}}

Each option is its own key with no separate description, so the model sees exactly the option text the compiler wrote.
A yes/no question is sent as a `noul` question with the model's own true/false wording and comes back as P(true). One
question per request: these models decide the questions of one request jointly, and the compiler's questions are meant
to be independent.
"""
import httpx

from s1agent.backends.base import Backend, Decision


class SystemOneLlamaCpp(Backend):
    max_options = 255

    def __init__(self, base_url, name=None, timeout=600.0):
        super().__init__()
        self.name = name or "systemone"
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=timeout)

    @staticmethod
    def request(state, question, options, kind="choice"):
        if kind == "noul":
            q = {"type": "noul", "instructions": question}
        else:
            q = {"type": "choice", "instructions": question, "criteria": {o: None for o in options}}
        return {"state": state, "questions": {"q": q}}

    def _choice(self, state, question, options, kind):
        r = self.http.post(f"{self.base_url}/v1/systemone", json=self.request(state, question, options, kind))
        if r.status_code != 200:
            raise RuntimeError(f"{self.name}: /v1/systemone returned {r.status_code}: {r.text[:300]}")
        body = r.json()
        ans = body["answers"]["q"]
        if kind == "noul":
            p = float(ans["noul"])
            probs = [1.0 - p, p]
        else:
            probs = [float(ans["probabilities"][o]) for o in options]
        return Decision(question, options, probs, kind, prompt_tokens=int(body.get("usage", {}).get("input_tokens", 0)),
                        meta={"backend": self.name, "model": body.get("model")})

    def health(self):
        return self.http.get(f"{self.base_url}/health").status_code == 200
