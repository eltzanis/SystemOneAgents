// Bridge from the scorer (s1eval.equivalence) to the ClinicalTrials.gov server's own input normalisers.
// Reads one JSON request per line on stdin, writes one JSON answer per line: {"out": <value or null>}.
//   {"fn": "status",  "value": <string or list>}            normalizeStatusFilter
//   {"fn": "phase",   "value": <string or list>}            normalizePhaseFilter
//   {"fn": "country", "value": <string>}                    registryCountry(queryTermContent(value))
//   {"fn": "state",   "value": <string>, "country": <str>}  registryState(queryTermContent(value), queryTermContent(country))
// The imported files are copied unchanged from clinicaltrialsgov-mcp-server 2.11.1 (see ../README.md).
import readline from 'node:readline';
import { normalizePhaseFilter, normalizeStatusFilter, queryTermContent } from './clinicaltrials/query-helpers.js';
import { registryCountry, registryState } from './clinicaltrials/place-names.js';

const FN = {
    status: (q) => normalizeStatusFilter(q.value) ?? null,
    phase: (q) => normalizePhaseFilter(q.value) ?? null,
    country: (q) => registryCountry(queryTermContent(q.value)),
    state: (q) => registryState(queryTermContent(q.value), queryTermContent(q.country ?? '')),
};

for await (const line of readline.createInterface({ input: process.stdin })) {
    let out = null;
    try {
        const q = JSON.parse(line);
        out = FN[q.fn](q);
    } catch {
        out = null;
    }
    process.stdout.write(JSON.stringify({ out }) + '\n');
}
