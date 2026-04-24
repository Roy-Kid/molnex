---
name: molnex-impl
description: Full implementation workflow from spec to production-ready code. Use when implementing new encoders, potentials, embeddings, interactions, or any non-trivial feature.
argument-hint: <feature description or spec path>
user-invocable: true
---

Implement the following feature in MolNex: $ARGUMENTS

**Execution discipline**: Before writing any code, enter **Plan Mode** to lay out the full plan, then create **Tasks** for each phase below. Update task status as work progresses (`in_progress` → `completed`). This enforces a structured, auditable workflow — the agent must not skip phases or jump ahead without completing prior tasks.

**Phase 1 — Literature Review** (for physical models only)
If the feature involves a physical model, potential, or operator, invoke `/molnex-litrev` first. Abort if no credible scientific basis is found.

**Phase 2 — Spec**
If `$ARGUMENTS` is a file path, read it. Otherwise invoke `/molnex-spec` to generate a detailed spec.

**Phase 3 — Architecture Design**
Use the `molnex-architect` agent to:
- Validate against the module dependency graph (molix.config ← molrep ← molzoo; molpot independent; molix.core ← application)
- Identify target package and affected modules
- Verify Pydantic config and cuEquivariance integration patterns
- Produce a module impact map

**Phase 4 — TDD (RED)**
Use the `molnex-tester` agent to write failing tests:
- Forward pass with correct tensor shapes
- Numerical validation against reference values (from literature)
- Symmetry tests: rotation, translation, permutation (where applicable)
- Edge cases: single atom, empty graph, large batch
- Run `python -m pytest tests/test_<pkg>/ -v` — confirm tests FAIL

**Phase 5 — Implement (GREEN)**
Write code following these rules:
- `torch>=2.6` APIs only
- `nn.Module` for neural network components; `BasePotential` for potentials
- `BaseModel` with `ConfigDict(arbitrary_types_allowed=True)` for configs
- cuEquivariance for tensor products; never manual einsum
- Accept/return plain dicts matching `MoleculeSample`/`MoleculeBatch` keys
- Functions < 50 lines, files < 800 lines, immutable data flow
- Paper reference in module docstring with arXiv/DOI
- Tensor shapes in all docstrings: ``(n_nodes, hidden_dim)``

Run `python -m pytest tests/test_<pkg>/ -v` — confirm tests PASS.

**Phase 6 — Review**
Run `/molnex-review` on all modified files.

**Phase 7 — Documentation**
Use the `molnex-documenter` agent to add Google-style docstrings, tensor shape annotations, Reference sections, and update `docs/` if needed.

**Phase 8 — Final Verification**
Run in parallel: `/molnex-arch`, `/molnex-test`, `/molnex-perf`, `/molnex-docs`.

Report: files created/modified, test results, coverage, literature references, remaining TODOs.
