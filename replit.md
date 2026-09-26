# Entity Resolution Pipeline

This workspace contains a standalone offline business-entity matching pipeline alongside the generated API and design scaffolds.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `cd business_entity_resolution && python run.py` — run entity resolution on local TSV files
- `cd business_entity_resolution && python -m unittest discover -s tests -v` — run Python unit tests
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Python 3.10+ for the standalone entity-resolution pipeline
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `business_entity_resolution/` — offline Python pipeline, tests, data contract, output validator, and methodology.
- `artifacts/api-server/` — generated Express API scaffold.
- `artifacts/mockup-sandbox/` — reusable component preview canvas.
- `lib/api-spec/` — OpenAPI source of truth for the generated API client.

## Architecture decisions

- Entity matching runs locally; it must not call commercial entity-resolution APIs, geocoding services, or cloud models.
- The Python pipeline uses two deterministic lexical blockers rather than downloading pretrained semantic model weights.
- The canonical output is long-form until the competition's official validator is available to define the exact submission schema.

## Product

The primary deliverable is a reproducible CPU-based matcher that generates candidate links, trains a precision-focused classifier, tunes a held-out threshold, and validates canonical outputs.

## User preferences

No project-specific preferences have been set.

## Gotchas

- Competition TSV data and the official output validator were not included with the build prompt; do not claim a valid leaderboard submission until both are supplied and checked.
- Run the Python pipeline from `business_entity_resolution/` so default relative paths resolve as documented.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
