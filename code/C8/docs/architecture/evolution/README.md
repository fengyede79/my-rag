# Runtime Architecture Evolution Specs

This folder breaks the frozen runtime architecture into small migration specs.

The frozen target architecture is:

- `../main-runtime-architecture-spec.md`

These evolution specs are intentionally staged. Each stage should produce its own implementation plan before code changes begin. Do not jump directly from the target architecture spec to a large rewrite.

## Stage Order

1. `00-current-state-and-migration-boundary.md`
2. `01-state-contract-and-writeback-policy.md`
3. `02-context-first-turn-pipeline.md`
4. `03-retrieval-executor-and-quality.md`
5. `04-context-packing-and-answer-modes.md`
6. `05-runtime-versioning-and-streaming.md`
7. `06-end-to-end-acceptance.md`

## Working Rule

For each stage:

1. Review the stage spec.
2. Expand details only for that stage.
3. Write an implementation plan for that stage.
4. Implement and test that stage.
5. Accept the stage before moving to the next one.

The purpose is controlled evolution, not a one-shot rewrite.

## Cutover Rule

The current runtime may fail to complete every target-architecture flow during migration, but it must not drift away from the frozen architecture.

When a new module takes ownership of an old responsibility, production calls must move to the new module. The old module may only remain as a thin wrapper if it is still actively called by the new path and has no independent behavior. Once the new module covers the old behavior, remove unused legacy functions, branches, nodes, paths, and tests.

Each implementation plan must explicitly state:

- which old responsibility is being replaced;
- where production calls will be cut over;
- which old code becomes illegal after cutover;
- which tests prove the new path replaced the old path;
- which legacy files or functions must be deleted before acceptance.
