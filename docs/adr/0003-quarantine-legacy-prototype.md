# 3. Quarantine the scripted prototype during the migration

Date: 2026-11-24 · Status: accepted (superseded when milestone 6 deletes `src/legacy/`)

## Context
Milestone 1 migrates the prototype to TypeScript and requires the app to keep running with the old scripted agent. The scripted pages, scenario data and demo store are fully replaced in milestone 6, so typing them precisely would be thrown-away work.

## Decision
Files that milestone 6 replaces live in `apps/web/src/legacy/` with `// @ts-nocheck` and are excluded from lint. Everything that survives (UI kit, layout primitives, StreamText, formatting helpers, providers) is fully typed under the spec's folder structure.

## Consequences
`tsc` is strict everywhere except `src/legacy/`. Milestone 6 must delete the folder; a leftover import would fail the build.
