# 4. Pin TypeScript 5.9 instead of the native TypeScript 7 compiler

Date: 2026-11-24 · Status: accepted

## Context
`npm install typescript` resolved to TypeScript 7 (the native Go compiler). It type-checks the app fine, but it does not expose the JavaScript compiler API (`ts.factory`) that `openapi-typescript` uses to generate `src/types/api.gen.ts`.

## Decision
Pin `typescript@~5.9` in `apps/web`. Revisit when openapi-typescript supports TypeScript 7.

## Consequences
`npm run gen-types` works; type-checking is slightly slower than with TS 7.
