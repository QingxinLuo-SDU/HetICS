import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.Objects

/-!
# CAVES-C strict stability filtering

The `RejectionFDRCertificate` records FDR control for erroneous rejections of stability nulls. It is
not selected-set FDR and does not establish simultaneous retention of a required core. Core retention
therefore remains the separate caller-supplied condition `CoreNotRejected`. The official fallback is
not modeled in this main interface.
-/

namespace BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC

set_option autoImplicit false
noncomputable section

variable {Feature : Type*}

/-- Candidate features rejected by the adjusted-p-value stability rule. -/
def bhRejected (candidates : Finset Feature)
    (adjustedP : Feature → ℝ) (alpha : ℝ) : Finset Feature :=
  candidates.filter fun j => adjustedP j < alpha

/-- Candidate features retained by the strict stability rule. -/
def strictKept (candidates : Finset Feature)
    (adjustedP : Feature → ℝ) (alpha : ℝ) : Finset Feature :=
  candidates.filter fun j => alpha ≤ adjustedP j

/-- Separate simultaneous condition that no required core feature is rejected. -/
def CoreNotRejected (core : Finset Feature)
    (adjustedP : Feature → ℝ) (alpha : ℝ) : Prop :=
  ∀ j ∈ core, alpha ≤ adjustedP j

/-- Certificate for rejection-FDR control, separate from any core-retention claim. -/
structure RejectionFDRCertificate where
  rejectionFdr : ℝ
  level : ℝ
  controlled : rejectionFdr ≤ level

/-- Every stability rejection came from the supplied candidate set. -/
theorem bhRejected_subset_candidates
    (candidates : Finset Feature) (adjustedP : Feature → ℝ) (alpha : ℝ) :
    bhRejected candidates adjustedP alpha ⊆ candidates := by
  intro j hj
  exact (Finset.mem_filter.mp hj).1

/-- Every strictly retained feature came from the supplied candidate set. -/
theorem strictKept_subset_candidates
    (candidates : Finset Feature) (adjustedP : Feature → ℝ) (alpha : ℝ) :
    strictKept candidates adjustedP alpha ⊆ candidates := by
  intro j hj
  exact (Finset.mem_filter.mp hj).1

/-- Candidate coverage and simultaneous non-rejection retain the entire required core. -/
theorem core_subset_strictKept
    (core candidates : Finset Feature) (adjustedP : Feature → ℝ) (alpha : ℝ)
    (hCandidates : core ⊆ candidates)
    (hNotRejected : CoreNotRejected core adjustedP alpha) :
    core ⊆ strictKept candidates adjustedP alpha := by
  intro j hj
  exact Finset.mem_filter.mpr ⟨hCandidates hj, hNotRejected j hj⟩

end

end BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC
