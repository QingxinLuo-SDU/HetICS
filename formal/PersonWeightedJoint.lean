import BinaryCore
import JointSandwich

/-! Integration only: reuse the checked first-batch sample partition.
No new count or weighted-mean foundation is proved here. -/
noncomputable section
open Matrix JointSandwich
open scoped BigOperators
namespace PersonWeightedJoint

variable {I K C D : Type*} [DecidableEq I] [DecidableEq K]
  [Fintype K] [DecidableEq C] [Fintype D]

/-- The joint inverse is instantiated with the actual nonempty-fiber person weights. -/
theorem person_weighted_inverse {P : K → Type*}
    [∀ k, Fintype (P k)] [∀ k, DecidableEq (P k)]
    (p : BinaryCore.SamplePartition I K)
    (A Q : ∀ k, Matrix (P k) (P k) ℝ)
    (hAQ : ∀ k, A k * Q k = 1) (hQA : ∀ k, Q k * A k = 1) :
    dependentBread p.weight A * dependentInverse p.weight Q = 1 ∧
    dependentInverse p.weight Q * dependentBread p.weight A = 1 := by
  have hw : ∀ k, p.weight k ≠ 0 := fun k => ne_of_gt (p.weight_pos k)
  exact ⟨dependent_inverse_right p.weight hw A Q hAQ,
    dependent_inverse_left p.weight hw A Q hQA⟩

/-- Center members are defined on precisely the first-batch sample set. -/
def centerRows (p : BinaryCore.SamplePartition I K) (center : I → C)
    (i : p.S) : Finset p.S :=
  Finset.univ.filter (fun j => center j = center i)

omit [Fintype K] in
theorem centerRows_nonempty (p : BinaryCore.SamplePartition I K) (center : I → C)
    (i : p.S) : (centerRows p center i).Nonempty := by
  exact ⟨i, by simp [centerRows]⟩

/-- N is the actual sample cardinality, a the actual fiber-count weights, and the
row domain is S itself. The transform R/H remains an explicit input, not a claim
about fitted derivatives or the Python array layout. -/
theorem person_weighted_centered_sandwich (p : BinaryCore.SamplePartition I K)
    (center : I → C) (R : Matrix D D ℝ) (H : Matrix K D ℝ)
    (u : p.S → D → ℝ) :
    p.weight ⬝ᵥ ((H * covariance (p.N : ℝ) R
      (fun i => u i - finiteMean (centerRows p center i) u) * H.transpose) *ᵥ p.weight) =
      ((p.N : ℝ) ^ 2)⁻¹ * ∑ i, (p.weight ⬝ᵥ (H *ᵥ (R *ᵥ u i)) -
        ((centerRows p center i).card : ℝ)⁻¹ *
          ∑ j ∈ centerRows p center i, p.weight ⬝ᵥ (H *ᵥ (R *ᵥ u j))) ^ 2 := by
  exact centered_sandwich_quadratic (p.N : ℝ) (by exact_mod_cast p.N_pos)
    R H p.weight u (centerRows p center)

#print axioms person_weighted_inverse
#print axioms centerRows_nonempty
#print axioms person_weighted_centered_sandwich
end PersonWeightedJoint
