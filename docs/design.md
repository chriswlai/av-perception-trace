# Learning Observable Perception–Reasoning Protocols with a Lightweight Understand–Decide–Verify Model
Author: Christopher Lai  
Target Domain: Autonomous Vehicle World Perception  
Primary Focus: Perception, Reasoning, Observability  
Final Goal: Lightweight LLM for structured Understand–Decide–Verify reasoning

## 1. Motivation
As autonomous driving systems increasingly incorporate learned models, observability and interpretability often degrade, especially when reasoning is embedded implicitly inside end-to-end policies or large opaque models.
In safety-critical systems, it is insufficient to know what action was taken. Engineers must understand:
- what the system believed about the world,
- which risks or constraints influenced behavior,
- and what conditions would have changed the decision.

This project reframes explainability as a perception–reasoning protocol problem, where reasoning is explicit, structured, learned, and auditable.
The final system uses a small, constrained language model to perform Understand–Decide–Verify reasoning over structured perception signals, producing explanations as a byproduct of reasoning, not post-hoc text generation.

## 2. Design Principles
- Perception is a contract, not an embedding.
- Reasoning must be inspectable.
- Learning must not destroy observability.
- Language is an interface, not a source of truth.
- Failure analysis is a first-class output.

## 3. System Overview
```
Visual Input
→ Explicit Perception Signals
→ TRACE Reasoning Protocol
→ Understand–Decide–Verify Model
→ Auditable Decision + Explanation
```

The system is organized around TRACE, a structured reasoning artifact that all models must produce.

## 4. TRACE: The Core Observability Protocol
TRACE Fields:
- T — Targets: relevant objects or regions (pedestrian_3, ego_lane, crosswalk)
- R — Relations: inter-object or object–ego relationships (crossing, following, merging, occluding)
- A — Action: STOP / SLOW / PROCEED with confidence
- C — Constraints / Risks: pedestrian_in_path, uncertain_object, traffic_light_transition
- E — Explanation: human-readable rendering derived from TRACE

TRACE as a Contract:
- All decision systems must output valid TRACE
- TRACE is logged, evaluated, and visualized
- Models are interchangeable as long as they honor TRACE

## 5. Perception Layer (Explicit, Non-Negotiable)
The reasoning model does not access raw pixels. It receives:
- object detections + confidence
- spatial context (ego lane, crosswalk)
- traffic light state
- distance estimates
- uncertainty flags

This enforces a clean perception → reasoning boundary.

## 6. Understand–Decide–Verify (UDV) Reasoning Model
The LLM is responsible for reasoning, not perception. It operates over structured perception inputs and produces structured TRACE outputs.

### 6.1 Understand
Produces a compact scene abstraction:
- relevant targets
- relations
- risks
- uncertainty summary

### 6.2 Decide
Selects:
- action (STOP / SLOW / PROCEED)
- action confidence
- active constraints

### 6.3 Verify
Performs self-checks:
- counterfactuals ("what would change this?")
- uncertainty awareness
- rule consistency

## 7. Training Strategy (Phased)
Phase 1 — Deterministic TRACE Generator  
Phase 2 — Learned Structured Reasoner (Non-LLM)  
Phase 3 — Lightweight LLM UDV Model

## 8. Explanation Generation (Derived, Not Free-Form)
Explanations are rendered from TRACE fields using templates for grounding and consistency.

## 9. Evaluation (Observability-First)
Metrics focus on action accuracy, factor precision/recall, relation correctness, and verification consistency.

## 10. Summary
The final system is a lightweight Understand–Decide–Verify reasoning model operating over structured perception protocols. By treating explanation as a contractual output of reasoning, the project remains interpretable, auditable, and safety-aware.
