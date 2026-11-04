import Mathlib.Analysis.InnerProductSpace.Projection.Basic

/-!
# Regression adjustment and the Frisch--Waugh--Lovell identity

The same real inner-product argument applies to finite observation vectors and to an `L2`
population space. The adjustment subspace contains center intercepts and selected covariates.
The FWL identity is derived from the regression normal equations. Identification then follows
from an additive-effect decomposition and either a linear outcome baseline or an exposure
conditional-mean projection that belongs to the adjustment space.

In the population interpretation, the larger subspace `G` consists of baseline-measurable
functions. Its orthogonal projection represents the exposure conditional mean. The embedding
of conditional expectation into this subspace and the causal/noise premises are statistical
inputs; they are not inferred from variable selection in this module.
-/

open scoped InnerProductSpace

namespace HetICS.RA

set_option autoImplicit false
noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Exposure residual after projection onto the center/covariate adjustment space. -/
def residual (B : Submodule ℝ E) [B.HasOrthogonalProjection] (A : E) : E :=
  A - B.starProjection A

/-- FWL ratio, with the usual nonzero residual-variation premise in its theorems. -/
def coefficient (B : Submodule ℝ E) [B.HasOrthogonalProjection] (A Y : E) : ℝ :=
  ⟪residual B A, Y⟫_ℝ / ⟪residual B A, residual B A⟫_ℝ

theorem residual_orthogonal (B : Submodule ℝ E) [B.HasOrthogonalProjection]
    (A v : E) (hv : v ∈ B) : ⟪residual B A, v⟫_ℝ = 0 := by
  exact B.starProjection_inner_eq_zero A v hv

/-- Residualizing the exposure does not change its inner product with that residual. -/
theorem residual_inner_exposure (B : Submodule ℝ E) [B.HasOrthogonalProjection]
    (A : E) : ⟪residual B A, A⟫_ℝ = ⟪residual B A, residual B A⟫_ℝ := by
  have hz := residual_orthogonal B A (B.starProjection A) (B.starProjection_apply_mem A)
  change ⟪residual B A, A⟫_ℝ = ⟪residual B A, A - B.starProjection A⟫_ℝ
  rw [inner_sub_right, hz, sub_zero]

/-- A coefficient satisfying the full regression normal equations equals the FWL ratio.
This is the finite-sample OLS bridge when `E` is a Euclidean observation space. -/
theorem fwl_of_normal_equations (B : Submodule ℝ E) [B.HasOrthogonalProjection]
    (A Y b : E) (theta : ℝ) (hb : b ∈ B)
    (hA : ⟪A, Y - (theta • A + b)⟫_ℝ = 0)
    (hB : ∀ v ∈ B, ⟪v, Y - (theta • A + b)⟫_ℝ = 0)
    (hVariation : ⟪residual B A, residual B A⟫_ℝ ≠ 0) :
    theta = coefficient B A Y := by
  have hp := hB (B.starProjection A) (B.starProjection_apply_mem A)
  have hr : ⟪residual B A, Y - (theta • A + b)⟫_ℝ = 0 := by
    change ⟪A - B.starProjection A, Y - (theta • A + b)⟫_ℝ = 0
    rw [inner_sub_left, hA, hp, sub_self]
  rw [inner_sub_right, inner_add_right, real_inner_smul_right,
    residual_orthogonal B A b hb, add_zero, residual_inner_exposure] at hr
  have hn : ⟪residual B A, Y⟫_ℝ = theta * ⟪residual B A, residual B A⟫_ℝ :=
    sub_eq_zero.mp hr
  unfold coefficient
  rw [hn, mul_div_cancel_right₀ _ hVariation]

/-- Exact sample error decomposition, without imposing sample noise orthogonality. -/
theorem coefficient_error_decomposition (B : Submodule ℝ E) [B.HasOrthogonalProjection]
    (A Y g error : E) (theta : ℝ)
    (hModel : Y = theta • A + g + error)
    (hVariation : ⟪residual B A, residual B A⟫_ℝ ≠ 0) :
    coefficient B A Y = theta +
      (⟪residual B A, g⟫_ℝ + ⟪residual B A, error⟫_ℝ) /
        ⟪residual B A, residual B A⟫_ℝ := by
  unfold coefficient
  rw [hModel, inner_add_right, inner_add_right, real_inner_smul_right,
    residual_inner_exposure, add_div, add_div,
    mul_div_cancel_right₀ _ hVariation, add_div, add_assoc]

/-- Population ratio bias after the noise-orthogonality premise is supplied. -/
theorem coefficient_bias_decomposition (B : Submodule ℝ E) [B.HasOrthogonalProjection]
    (A Y g error : E) (theta : ℝ)
    (hModel : Y = theta • A + g + error)
    (hNoise : ⟪residual B A, error⟫_ℝ = 0)
    (hVariation : ⟪residual B A, residual B A⟫_ℝ ≠ 0) :
    coefficient B A Y = theta +
      ⟪residual B A, g⟫_ℝ / ⟪residual B A, residual B A⟫_ℝ := by
  unfold coefficient
  rw [hModel, inner_add_right, inner_add_right, real_inner_smul_right,
    hNoise, add_zero, residual_inner_exposure, add_div,
    mul_div_cancel_right₀ _ hVariation]

/-- Correct linear outcome baseline: exposure-mean linearity is not required. -/
theorem coefficient_eq_of_outcome_linear (B : Submodule ℝ E) [B.HasOrthogonalProjection]
    (A Y g error : E) (theta : ℝ)
    (hModel : Y = theta • A + g + error) (hg : g ∈ B)
    (hNoise : ⟪residual B A, error⟫_ℝ = 0)
    (hVariation : ⟪residual B A, residual B A⟫_ℝ ≠ 0) :
    coefficient B A Y = theta := by
  rw [coefficient_bias_decomposition B A Y g error theta hModel hNoise hVariation,
    residual_orthogonal B A g hg, zero_div, add_zero]

/-- If the baseline-function projection of exposure belongs to the adjustment space,
then the two projections agree. This expresses correct exposure-mean specification. -/
theorem exposure_projections_eq (B G : Submodule ℝ E)
    [B.HasOrthogonalProjection] [G.HasOrthogonalProjection]
    (A : E) (hBG : B ≤ G) (hMean : G.starProjection A ∈ B) :
    B.starProjection A = G.starProjection A := by
  apply B.eq_starProjection_of_mem_of_inner_eq_zero hMean
  intro v hv
  exact G.starProjection_inner_eq_zero A v (hBG hv)

/-- Correct exposure conditional mean: the baseline may be nonlinear in the selected covariates,
provided it belongs to the larger baseline-function space `G`. -/
theorem coefficient_eq_of_exposure_linear (B G : Submodule ℝ E)
    [B.HasOrthogonalProjection] [G.HasOrthogonalProjection]
    (A Y g error : E) (theta : ℝ)
    (hBG : B ≤ G) (hMean : G.starProjection A ∈ B) (hg : g ∈ G)
    (hModel : Y = theta • A + g + error)
    (hNoise : ⟪residual B A, error⟫_ℝ = 0)
    (hVariation : ⟪residual B A, residual B A⟫_ℝ ≠ 0) :
    coefficient B A Y = theta := by
  have hOrth : ⟪residual B A, g⟫_ℝ = 0 := by
    unfold residual
    rw [exposure_projections_eq B G A hBG hMean]
    exact G.starProjection_inner_eq_zero A g hg
  rw [coefficient_bias_decomposition B A Y g error theta hModel hNoise hVariation,
    hOrth, zero_div, add_zero]

/-- Either correctly specified working component identifies the same exposure coefficient. -/
theorem coefficient_eq_of_either_linear (B G : Submodule ℝ E)
    [B.HasOrthogonalProjection] [G.HasOrthogonalProjection]
    (A Y g error : E) (theta : ℝ) (hBG : B ≤ G) (hg : g ∈ G)
    (hModel : Y = theta • A + g + error)
    (hNoise : ⟪residual B A, error⟫_ℝ = 0)
    (hVariation : ⟪residual B A, residual B A⟫_ℝ ≠ 0)
    (hEither : G.starProjection A ∈ B ∨ g ∈ B) :
    coefficient B A Y = theta := by
  rcases hEither with hMean | hOutcome
  · exact coefficient_eq_of_exposure_linear B G A Y g error theta hBG hMean hg
      hModel hNoise hVariation
  · exact coefficient_eq_of_outcome_linear B A Y g error theta hModel hOutcome
      hNoise hVariation

end
end HetICS.RA
