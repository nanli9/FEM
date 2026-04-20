---
name: run-fem-milestone
description: Execute one FEM milestone with tests, demo commands, and minimal coherent edits.
disable-model-invocation: true
allowed-tools: Read Write Edit MultiEdit Glob Grep Bash(pytest *) Bash(python scripts/*)
---

Execute FEM milestone: $ARGUMENTS

Rules:
1. Identify the exact milestone target.
2. Check prerequisite modules first.
3. Make the smallest coherent set of changes.
4. Add or update tests.
5. Add or update one runnable demo.
6. Run the relevant tests and demo command.
7. Report:
   - files changed
   - tests run
   - demo command
   - known gaps
