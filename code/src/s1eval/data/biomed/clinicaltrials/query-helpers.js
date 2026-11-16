/**
 * @fileoverview Shared helpers for normalizing tool inputs into API search parameters.
 * @module mcp-server/tools/utils/query-helpers
 */
export function toArray(v) {
    if (v === undefined)
        return;
    if (Array.isArray(v))
        return v;
    const trimmed = v.trim();
    if (trimmed.startsWith('[')) {
        try {
            const parsed = JSON.parse(trimmed);
            if (Array.isArray(parsed) && parsed.every((x) => typeof x === 'string')) {
                return parsed;
            }
        }
        catch {
            // Not valid JSON — fall through to the scalar-wrap (unchanged behavior).
        }
    }
    return [v];
}
/**
 * Normalize a `statusFilter` to upstream's canonical `OverallStatus` spelling.
 *
 * `filter.overallStatus` is matched case-sensitively upstream — `recruiting`
 * and `Recruiting` both 400 while `RECRUITING` succeeds — so each entry is
 * trimmed, uppercased, and every run of whitespace, hyphens, and underscores
 * collapsed to one `_`. No two statuses differ only in case or separators, so
 * the mapping is one-to-one. Whitespace around a `,` or `|` list delimiter
 * (both of which upstream splits on) is dropped first, so `recruiting, completed`
 * becomes `RECRUITING,COMPLETED` rather than a `_COMPLETED` token the caller
 * never wrote. The two registry display labels that differ from their API value
 * by more than case and separators, `Active, not recruiting` (its comma would
 * split it) and `Unknown status`, are mapped to `ACTIVE_NOT_RECRUITING` and
 * `UNKNOWN`; neither `ACTIVE`, `NOT_RECRUITING`, nor `UNKNOWN_STATUS` is a
 * status, so the mapping stays one-to-one. A value that still names no status
 * reaches upstream in its normalized form and fails there with `enum_invalid`,
 * the same contract and recovery hint an unknown canonical value gets. A blank
 * entry becomes `''`, which `firstBlankListParam` still rejects.
 */
export function normalizeStatusFilter(v) {
    return toArray(v)?.map((s) => s
        .trim()
        .toUpperCase()
        .replace(/\s*([,|])\s*/g, '$1')
        .replace(/[\s_-]+/g, '_')
        .replace(/(^|[,|])ACTIVE,NOT_RECRUITING(?=$|[,|])/g, '$1ACTIVE_NOT_RECRUITING')
        .replace(/(^|[,|])UNKNOWN_STATUS(?=$|[,|])/g, '$1UNKNOWN'));
}
/**
 * Unicode format characters (`\p{Cf}`: zero-width space U+200B, word joiner
 * U+2060, BOM U+FEFF, …). Invisible, and `String.trim()` keeps most of them, so
 * a value made of them looks blank to a caller yet passes a trim-based check.
 */
const FORMAT_CHARS = /\p{Cf}/gu;
/** A value with its format characters dropped and its whitespace trimmed. */
function visibleText(value) {
    return value.replace(FORMAT_CHARS, '').trim();
}
export function splitListEntries(values) {
    if (!values)
        return;
    const pieces = values.flatMap((entry) => {
        const split = entry
            .split(/[,|]/)
            .map((p) => p.trim())
            .filter((p) => visibleText(p).length > 0);
        return split.length > 0 ? split : [''];
    });
    return [...new Set(pieces)];
}
/** The six `Phase` values ClinicalTrials.gov accepts, in registry order. */
export const PHASE_VALUES = ['NA', 'EARLY_PHASE1', 'PHASE1', 'PHASE2', 'PHASE3', 'PHASE4'];
/**
 * Canonical phase keyed by its spelling with case, whitespace, `-`, `_`, and
 * quotes removed. `NOTAPPLICABLE` is the registry display label of `NA`.
 */
const PHASE_BY_KEY = new Map([
    ...PHASE_VALUES.map((p) => [p.replaceAll('_', ''), p]),
    ['NOTAPPLICABLE', 'NA'],
]);
/**
 * Resolve a `phaseFilter` to canonical phase values.
 *
 * Each entry is split on `,` and `|` (`splitListEntries`), then matched with
 * case, whitespace, `-`, `_`, and quotes ignored, so every single-value
 * spelling upstream accepts (`phase3`, `Phase 3`, `"PHASE3"`, `Not Applicable`,
 * `Early Phase 1`) resolves to the one value it names. A piece that resolves to
 * no phase is returned as the whole entry it came from (trimmed) for
 * `firstInvalidPhase` to name — `RANGE[PHASE1,PHASE3]` is named as written, not
 * as the fragment `RANGE[PHASE1`. Entries are interpolated into
 * `AREA[Phase]<entry>`, so one such as `PHASE3 OR AREA[StudyType]OBSERVATIONAL`
 * must never reach upstream, which would answer it as an expression widening
 * the cohort past the phase constraint. Quotes are ignored in matching, so a
 * piece of only quotes, whitespace, and format characters carries no content:
 * it is dropped like an empty piece, and an entry left with nothing stays `''`
 * for `firstBlankListParam`.
 */
export function normalizePhaseFilter(v) {
    return toArray(v)?.flatMap((entry) => {
        const pieces = splitListEntries([entry]).filter((p) => visibleText(p.replace(/["']/g, '')).length > 0);
        if (pieces.length === 0)
            return [''];
        return pieces.map((p) => PHASE_BY_KEY.get(p.toUpperCase().replace(/[\s_"'-]/g, '')) ?? entry.trim());
    });
}
/** The first entry of a normalized `phaseFilter` that is not a canonical phase. */
export function firstInvalidPhase(phases) {
    return phases?.find((p) => !PHASE_VALUES.includes(p));
}
/** Message and recovery hint for a `phaseFilter` entry that names no phase. */
export function invalidPhaseRejection(value) {
    const values = PHASE_VALUES.join(', ');
    return {
        message: `Invalid value for \`phaseFilter\`: '${value}' is not a trial phase. Each phaseFilter entry must be one of ${values} — search expressions are not accepted here.`,
        hint: `Use one of ${values} per phaseFilter entry, and pass a list such as ["PHASE2","PHASE3"] to match any of several phases. Put every other criterion, including OR/NOT expressions, in advancedFilter.`,
    };
}
/**
 * Name the first supplied-but-blank string parameter, or `undefined` when every
 * supplied value carries non-whitespace. Omitted parameters are skipped —
 * omission keeps its documented meaning; only a value the caller actually
 * supplied is judged. Trimming, not a length check: `' '` is as empty as `''`,
 * and the upstream query builder drops both, silently widening the request to
 * the whole registry instead of constraining it.
 */
export function firstBlankParam(params) {
    for (const [name, value] of Object.entries(params)) {
        if (value !== undefined && value.trim().length === 0)
            return name;
    }
    return undefined;
}
/**
 * Name the first supplied list parameter that is empty or carries a blank
 * entry. Pass lists already normalized through `toArray`, so the stringified
 * `'[]'` form LLM callers sometimes send is judged on what it resolves to.
 */
export function firstBlankListParam(params) {
    for (const [name, values] of Object.entries(params)) {
        if (!values)
            continue;
        if (values.length === 0 || values.some((v) => v.trim().length === 0))
            return name;
    }
    return undefined;
}
/**
 * Actionable message for a parameter supplied with a blank value. Leads with
 * the fix that works on a required parameter too — omitting one of those fails
 * the schema and returns a bare -32602 with no reason and no recovery hint.
 */
export function blankValueMessage(param) {
    return `Parameter '${param}' was supplied with a blank value — an empty or whitespace-only string, an empty list, or a list carrying a blank entry. Supply a value containing non-whitespace, or for a list at least one non-blank entry; omit the parameter entirely only if it is optional and you meant to leave it unset.`;
}
/**
 * Build the AREA[] phase filter and combine it with the caller's advancedFilter.
 *
 * The caller's expression is parenthesized as a unit before the AND join. Essie
 * has no precedence rule that scopes a trailing `OR` back under a preceding
 * `AND`, so an ungrouped operand let the OR branch escape the phase constraint
 * entirely — a PHASE3 search answered with phase-less observational studies.
 * The wrap is unconditional: Essie tolerates redundant nested parentheses, and
 * deciding whether a caller's own leading `(` closes at the end of the
 * expression or midway through it would mean reimplementing the parser.
 *
 * An advancedFilter with no phase expression alongside it is returned untouched
 * — there is no AND boundary for its meaning to leak past.
 */
export function buildAdvancedFilter(phaseFilter, advancedFilter) {
    const phaseExpr = phaseFilter?.length
        ? phaseFilter.length === 1
            ? `AREA[Phase]${phaseFilter[0]}`
            : `(${phaseFilter.map((p) => `AREA[Phase]${p}`).join(' OR ')})`
        : undefined;
    if (!advancedFilter)
        return phaseExpr;
    return phaseExpr ? `${phaseExpr} AND (${advancedFilter})` : advancedFilter;
}
/**
 * Render one structured term as an upstream query operand: quoted when it
 * carries whitespace so it matches as a literal phrase, bare otherwise.
 *
 * An embedded `"` is stripped rather than escaped. Upstream Essie has no
 * working escape for a literal quote inside a quoted phrase and does not fail
 * on one — an unescaped `"` silently reparses into a different query that
 * returns a plausible-looking wrong result set, and a backslash-escaped one
 * matches nothing. Removing the character is the only handling that keeps the
 * phrase intact.
 */
export function quoteQueryTerm(term) {
    const sanitized = term.replaceAll('"', '');
    return /\s/.test(sanitized) ? `"${sanitized}"` : sanitized;
}
/**
 * Render a location search term (the city) as an operand inside
 * `SEARCH[Location](…)`: bare when it is a single run of letters and digits,
 * otherwise quoted as `quoteLocationLiteral` quotes it. Inside the wrapper a
 * bare part carrying punctuation under-matches — `Winston-Salem` finds a small
 * fraction of the studies its quoted form finds, every one of which lists a
 * Winston-Salem site — and a bare `(`, `)`, `[`, or `]` is query syntax that
 * would close the wrapper and regroup the expression around the rest of the
 * part. An embedded `"` is stripped before deciding.
 */
export function quoteLocationPart(part) {
    const sanitized = part.replaceAll('"', '');
    return /[^\p{L}\p{N}]/u.test(sanitized) ? quoteLocationLiteral(sanitized) : sanitized;
}
/**
 * Render a location value as one quoted literal, the form an
 * `AREA[LocationState]` or `AREA[LocationCountry]` operand always takes: left
 * bare there, a single word is read as syntax — `Missing` as the `MISSING`
 * operator. An embedded `"` is stripped, as in `quoteQueryTerm`. A backslash is
 * doubled: a single one escapes the closing quote, leaving the phrase open so a
 * later part's `)` regroups the expression, while upstream reads a doubled one
 * inside a phrase as a literal backslash.
 */
export function quoteLocationLiteral(value) {
    return `"${value.replaceAll('"', '').replaceAll('\\', '\\\\')}"`;
}
/**
 * What a term contributes to the query `quoteQueryTerm`, `quoteLocationPart`,
 * or `quoteLocationLiteral` builds: its text with every `"` and every invisible
 * format character stripped, trimmed. Empty means the term carries no content —
 * it reaches upstream as no operand at all, as a bare `" "` phrase, or as a
 * zero-width term upstream answers with a bare `Search error` — so judge
 * blankness on this, not on the raw value.
 */
export function queryTermContent(term) {
    return visibleText(term.replaceAll('"', ''));
}
/**
 * Report normalized filter values in the shape the caller sent them: a list
 * stays a list, and a lone scalar comes back as a scalar.
 */
export function inCallerShape(raw, normalized) {
    return Array.isArray(raw) || normalized.length !== 1 ? normalized : normalized[0];
}
//# sourceMappingURL=query-helpers.js.map