## Next Steps Checklist (V2.1)

- [x] Phase 1: Expand deterministic TRACE rules (traffic lights, stop lines, lanes, uncertainty).
- [x] Phase 1: Add richer perception contract tests and schema validation.
- [x] Phase 1: Extend explanation templates for new constraint types.
- [x] Phase 1: Add report section summarizing verification and constraints stats.
- [x] Phase 1: Add failure taxonomy tags to TRACE records.

- [x] Phase 2: Implement a small supervised model to predict action/constraints/relations.
- [x] Phase 2: Train on teacher traces and compare against baseline rules.
- [x] Phase 2: Add evaluation metrics (action accuracy, factor precision/recall).

- [x] Phase 3: Build UDV prompt + schema validation loop.
- [x] Phase 3: Add counterfactual checks and verification consistency scoring.
- [x] Phase 3: Curate a small TRACE/UDV example set for regression tests.

- [x] Pipeline: Add a one-shot `run_all.py` flow to regenerate frames/traces/report.
- [ ] Pipeline: Add report flags for optional sections (map, CAN bus, verify).
- [ ] Docs: Update README with full flow + report interpretation guide.
