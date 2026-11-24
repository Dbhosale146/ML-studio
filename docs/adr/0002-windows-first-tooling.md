# 2. Windows-friendly developer tooling (Makefile + make.ps1)

Date: 2026-11-24 · Status: accepted

## Context
The spec standardises on `make` and Docker. The first development machine runs Windows 11 without GNU make or Docker installed; it does have Git, Node 24 and uv.

## Decision
- The `Makefile` stays the canonical command list (used by CI and macOS/Linux developers).
- `make.ps1` mirrors every target one-to-one for Windows (`./make.ps1 dev`, `./make.ps1 test`, ...).
- Dockerfiles and `docker-compose.yml` are provided and kept in sync, but are not exercised on the Windows dev machine; CI (Linux) is the place they get built.

## Consequences
Two command files must be kept in sync. Docker changes need to be validated in CI or on a machine with Docker.
