import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.Sigma
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring

/-!
# Whole-center cross-fitted PLR interface for CAVES-C

The split object assigns folds to whole centers; it deliberately does not expose an unconstrained
row-level `Observation → Fold` map. This formal split governs only PLR nuisance fitting. It does not
repair same-sample upstream sEAM, Y-side screening, or Chow post-selection validity.

The score results below are finite deterministic algebra, not a central limit theorem or a coverage
theorem. Chiang--Kato--Ma--Sasaki, Appendix C, in the one-way `ell = 1` case is the external boundary
for the center-clustered CLT and variance result. In particular, this module does not prove that
`RidgeCV` nuisance fits satisfy the named rate, regime, CLT, or covariance-consistency conditions.
-/

open scoped BigOperators

namespace BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC

set_option autoImplicit false
noncomputable section

variable {Center Fold Observation : Type*}

/-- A PLR fold assignment whose atomic splitting unit is an entire center. -/
structure WholeCenterSplit (Center Fold : Type*) where
  foldOfCenter : Center → Fold

/-- An observation inherits its fold only through its center. -/
def observationFold (split : WholeCenterSplit Center Fold)
    (centerOf : Observation → Center) (i : Observation) : Fold :=
  split.foldOfCenter (centerOf i)

/-- Centers available for nuisance training when fold `f` is held out. -/
def trainingCenters [Fintype Center] [DecidableEq Fold]
    (split : WholeCenterSplit Center Fold) (f : Fold) : Finset Center :=
  Finset.univ.filter fun c => split.foldOfCenter c ≠ f

/-- Whole centers held out for validation in fold `f`. -/
def validationCenters [Fintype Center] [DecidableEq Fold]
    (split : WholeCenterSplit Center Fold) (f : Fold) : Finset Center :=
  Finset.univ.filter fun c => split.foldOfCenter c = f

/-- Observations from the same center necessarily inherit the same PLR fold. -/
theorem observationFold_eq_of_same_center
    (split : WholeCenterSplit Center Fold) (centerOf : Observation → Center)
    (i j : Observation) (hSameCenter : centerOf i = centerOf j) :
    observationFold split centerOf i = observationFold split centerOf j := by
  exact congrArg split.foldOfCenter hSameCenter

/-- For one fold, no center is both a nuisance-training and validation center. -/
theorem trainingCenters_disjoint_validationCenters
    [Fintype Center] [DecidableEq Fold]
    (split : WholeCenterSplit Center Fold) (f : Fold) :
    Disjoint (trainingCenters split f) (validationCenters split f) := by
  classical
  refine Finset.disjoint_left.mpr ?_
  intro c hTraining hValidation
  simp only [trainingCenters, Finset.mem_filter, Finset.mem_univ, true_and] at hTraining
  simp only [validationCenters, Finset.mem_filter, Finset.mem_univ, true_and] at hValidation
  exact hTraining hValidation

/-- Training membership is exactly the complement of validation membership for the same fold. -/
theorem center_mem_training_iff_not_mem_validation
    [Fintype Center] [DecidableEq Fold]
    (split : WholeCenterSplit Center Fold) (f : Fold) (c : Center) :
    c ∈ trainingCenters split f ↔ c ∉ validationCenters split f := by
  classical
  simp [trainingCenters, validationCenters]

section FinitePLRAlgebra

variable {Center : Type*} [Fintype Center]
variable {Unit : Center → Type*} [∀ c, Fintype (Unit c)]

/-- The unnormalised PLR score contributed by center `c`. -/
def centerScore (c : Center) (v u : Sigma Unit → ℝ) : ℝ :=
  ∑ i : Unit c, v ⟨c, i⟩ * u ⟨c, i⟩

/-- The unnormalised PLR score over all center-observation pairs. -/
def pooledScore (v u : Sigma Unit → ℝ) : ℝ :=
  ∑ i, v i * u i

/-- The finite residual-on-residual PLR ratio. -/
def clusterEffectEstimate (v yResidual : Sigma Unit → ℝ) : ℝ :=
  pooledScore v yResidual / pooledScore v v

/-- The sum of squared center score contributions used in a one-way sandwich numerator. -/
def centerSandwichNumerator (v residual : Sigma Unit → ℝ) : ℝ :=
  ∑ c, centerScore c v residual ^ 2

/-- The pooled finite score is exactly the sum of the center scores. -/
theorem pooledScore_eq_sum_centerScore (v u : Sigma Unit → ℝ) :
    pooledScore v u = ∑ c, centerScore c v u := by
  classical
  simp only [pooledScore, centerScore]
  rw [← Finset.univ_sigma_univ, Finset.sum_sigma]

/-- Centerwise zero scores imply an exactly zero pooled finite score. -/
theorem pooledScore_eq_zero_of_centerScore_eq_zero
    (v u : Sigma Unit → ℝ) (hCenterScore : ∀ c, centerScore c v u = 0) :
    pooledScore v u = 0 := by
  rw [pooledScore_eq_sum_centerScore]
  apply Finset.sum_eq_zero
  intro c hc
  exact hCenterScore c

/--
Under an exact linear residual identity, pooled score orthogonality and nonzero residual-treatment
variation make the finite PLR ratio equal `θ`. This is deterministic algebra, not asymptotic or
coverage validity.
-/
theorem clusterEffectEstimate_eq
    (θ : ℝ) (v yResidual u : Sigma Unit → ℝ)
    (hResidual : ∀ i, yResidual i = θ * v i + u i)
    (hOrthogonal : pooledScore v u = 0)
    (hVariation : pooledScore v v ≠ 0) :
    clusterEffectEstimate v yResidual = θ := by
  have hNumerator : pooledScore v yResidual = θ * pooledScore v v := by
    calc
      pooledScore v yResidual = pooledScore v (fun i => θ * v i + u i) := by
        simp only [pooledScore]
        apply Finset.sum_congr rfl
        intro i hi
        rw [hResidual i]
      _ = θ * pooledScore v v := by
        have hOrthogonal' : (∑ i, v i * u i) = 0 := by
          simpa only [pooledScore] using hOrthogonal
        simp only [pooledScore, mul_add, Finset.sum_add_distrib]
        rw [hOrthogonal']
        simp only [add_zero]
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro i hi
        ring
  simp only [clusterEffectEstimate]
  rw [hNumerator]
  rw [mul_comm θ (pooledScore v v)]
  exact mul_div_cancel_left₀ θ hVariation

/-- A finite sum of squared center scores is nonnegative. -/
theorem centerSandwichNumerator_nonneg (v residual : Sigma Unit → ℝ) :
    0 ≤ centerSandwichNumerator v residual := by
  simp only [centerSandwichNumerator]
  exact Finset.sum_nonneg fun c hc => sq_nonneg (centerScore c v residual)

end FinitePLRAlgebra

/--
Named assumptions supplied by a caller at the external PLR source-theorem boundary. Chiang et al.,
Appendix C with one-way `ell = 1`, supplies the relevant CLT/variance target; no field is proved here,
and no claim is made that `RidgeCV` establishes any field.
-/
structure ClusterPLRSourceConditions where
  nuisanceProductRate : Prop
  centerAsymptoticRegime : Prop
  centerCLT : Prop
  covarianceConsistent : Prop

end

end BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC
