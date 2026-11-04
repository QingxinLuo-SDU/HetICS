# HetICS formal proofs

Lean 4 sources for regression adjustment (the primary continuous estimator), the binary AIPW branch, and the earlier Robinson comparison estimator.

## Build

The package pins Lean **4.32.0** and Mathlib commit **81a5d257c8e410db227a6665ed08f64fea08e997**. With [elan](https://github.com/leanprover/elan) installed:

```sh
cd formal
lake exe cache get
lake build
lake env lean Check.lean
```

Dependencies retain their own licenses. Compiled objects and dependency caches are not included.

## RA proof chain

| Module | Checked result | Statistical inputs |
|---|---|---|
| `HetICS/RAProjection.lean` | FWL from the full regression normal equations; exact residual error decomposition; identification when either working component is linear | An adjustment subspace, additive causal model and noise orthogonality; conditional expectation represented by projection onto the baseline-function space |
| `HetICS/RAMoments.lean` | The residual ratio is consistent when its numerator vanishes and its denominator has a positive limit; deterministic center-score gap preservation | Empirical moment convergence supplied by the sampling assumptions; population separation and score-approximation bounds |
| `HetICS/RASelection.lean` | Dual-side union and threshold-parametric retention; same-data finite-family consistency; actual integral L1 convergence under random convex aggregation | A high-probability event on which the selected estimator belongs to the finite admissible reference family; reference moment conditions and measurable weights |
| `HetICS/RAExpectation.lean` | Mathlib Vitali theorem gives actual absolute-error integral and expectation convergence, including an integrable random target | Convergence in measure and uniform integrability of the complete estimator |
| `HetICS/RAPipeline.lean` | `learned_ra_point_guarantee` combines moment convergence, finite-family learning and Vitali into consistency, L1 consistency and asymptotic unbiasedness; `common_effect_aggregate_unbiased` covers convex-weight aggregation | The FWL ratio representation on the admissibility event, and the explicitly listed premises of the component results |

The expectation conclusions use the actual Bochner integral of the estimator. A vanishing bias sequence is not assumed in place of proving this conclusion. Failed computations can be represented by a fixed finite mathematical extension; the statistical availability and integrability conditions concern that complete estimator.

The invariance threshold is an argument `alpha : Real`. The experimental value `0.001` is not built into the proof. A fixed, sample-size-dependent, or data-selected threshold is represented through the final selected set and its admissibility event. The retention results state the required coverage/nonrejection premises rather than deriving causal sufficiency from a significance threshold.

## Existing continuous and binary results

`BiostatFormal/HighDimensional/ContinuousPipeline/CAVESC/` retains the earlier label, selection, Robinson, GLS and inference interfaces. `BinarySubmission.lean` is the assembled binary entry point; its dependencies include AIPW bias identities, actual logistic score/Jacobian calculations, person-weighted aggregation, finite-center estimating equations and learning-rule transfer. These files are included with their transitive project dependencies and original theorem statements.

The manuscript's assumptions and the formal theorem premises have distinct roles. The new RA point-estimation chain is conditional on moment convergence, successful learning into an admissible family, and uniform integrability. Raw-data laws of large numbers, convergence of the finite-iteration clustering optimizer, the full heterogeneous-REML L1 theorem, and the new RA center-CLT/CR1 argument in Supplementary S3.6 remain manuscript-level arguments. The earlier PLR inference modules concern the Robinson estimator. They are not substituted for RA interval theory.

`Check.lean` imports both branches and prints the axiom dependencies of representative new and reused theorems. The source package contains no proof placeholders or project-defined statistical axioms. Ordinary Lean foundational axioms are reported by the audit.
