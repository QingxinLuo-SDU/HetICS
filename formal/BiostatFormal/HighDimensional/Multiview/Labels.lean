import Mathlib.Data.Fintype.Perm
import Mathlib.Data.Finset.Max
import Mathlib.InformationTheory.Hamming

/-!
# Labels modulo a global permutation

This file formalizes the label-space part of the multiview Bernoulli-sEAM foundations.  Labels
remain an arbitrary finite type: neither their names nor an ordering on them is statistically
meaningful.  A single permutation acts at every center, which is the source model's global label
switching convention.
-/

namespace BiostatFormal.HighDimensional.Multiview

universe u v

/-- Informal `z ∈ {1, ..., K}^E` ↔ Lean: a function from centers to a finite label type. -/
abbrev Partition (Center : Type u) (Label : Type v) := Center → Label

/-- Informal `φ ∈ S_K` ↔ Lean: a permutation of the finite label type. -/
abbrev LabelPermutation (Label : Type v) := Equiv.Perm Label

/-- Informal `φ(z)` ↔ Lean: apply the same label permutation at every center. -/
def relabel {Center : Type u} {Label : Type v}
    (σ : LabelPermutation Label) (z : Partition Center Label) : Partition Center Label :=
  fun center => σ (z center)

@[simp]
theorem relabel_apply {Center : Type u} {Label : Type v}
    (σ : LabelPermutation Label) (z : Partition Center Label) (center : Center) :
    relabel σ z center = σ (z center) :=
  rfl

@[simp]
theorem relabel_one {Center : Type u} {Label : Type v} (z : Partition Center Label) :
    relabel (1 : LabelPermutation Label) z = z := by
  rfl

theorem relabel_trans {Center : Type u} {Label : Type v}
    (σ τ : LabelPermutation Label) (z : Partition Center Label) :
    relabel τ (relabel σ z) = relabel (σ.trans τ) z := by
  rfl

/-- Informal `[z] = [z']` ↔ Lean: one global permutation sends `z` to `z'`. -/
def LabelEquivalent {Center : Type u} {Label : Type v}
    (z z' : Partition Center Label) : Prop :=
  ∃ σ : LabelPermutation Label, relabel σ z = z'

theorem labelEquivalent_refl {Center : Type u} {Label : Type v}
    (z : Partition Center Label) : LabelEquivalent z z :=
  ⟨1, relabel_one z⟩

theorem labelEquivalent_symm {Center : Type u} {Label : Type v}
    {z z' : Partition Center Label} (h : LabelEquivalent z z') : LabelEquivalent z' z := by
  rcases h with ⟨σ, hσ⟩
  refine ⟨σ.symm, ?_⟩
  funext center
  have hAt := congrFun hσ center
  simpa using (congrArg σ.symm hAt).symm

theorem labelEquivalent_trans {Center : Type u} {Label : Type v}
    {z₁ z₂ z₃ : Partition Center Label}
    (h₁₂ : LabelEquivalent z₁ z₂) (h₂₃ : LabelEquivalent z₂ z₃) :
    LabelEquivalent z₁ z₃ := by
  rcases h₁₂ with ⟨σ, hσ⟩
  rcases h₂₃ with ⟨τ, hτ⟩
  refine ⟨σ.trans τ, ?_⟩
  rw [← relabel_trans, hσ, hτ]

/-- Informal label-switching equivalence ↔ Lean: a setoid on partitions. -/
def labelEquivalentSetoid (Center : Type u) (Label : Type v) :
    Setoid (Partition Center Label) where
  r := LabelEquivalent
  iseqv := ⟨labelEquivalent_refl, labelEquivalent_symm, labelEquivalent_trans⟩

/-- Informal `[z]` ↔ Lean: a quotient class of partitions modulo global relabeling. -/
abbrev PartitionOrbit (Center : Type u) (Label : Type v) :=
  Quotient (labelEquivalentSetoid Center Label)

/-- Hamming mismatch count after applying one specified global permutation. -/
def permutationMismatchCount {Center : Type u} {Label : Type v}
    [Fintype Center] [DecidableEq Center] [DecidableEq Label]
    (z z' : Partition Center Label) (σ : LabelPermutation Label) : ℕ :=
  hammingDist (relabel σ z) z'

/-- Informal unnormalized numerator of `d_Z` ↔ Lean: the minimum finite Hamming mismatch count. -/
noncomputable def optimalMismatchCount {Center : Type u} {Label : Type v}
    [Fintype Center] [DecidableEq Center] [Fintype Label] [DecidableEq Label]
    (z z' : Partition Center Label) : ℕ :=
  let counts := Finset.univ.image (permutationMismatchCount z z')
  counts.min' (by
    refine ⟨permutationMismatchCount z z' (1 : LabelPermutation Label), ?_⟩
    simp [counts])

theorem labelEquivalent_iff_optimalMismatchCount_eq_zero
    {Center : Type u} {Label : Type v}
    [Fintype Center] [DecidableEq Center] [Fintype Label] [DecidableEq Label]
    {z z' : Partition Center Label} :
    LabelEquivalent z z' ↔ optimalMismatchCount z z' = 0 := by
  constructor
  · rintro ⟨σ, hσ⟩
    have hLe : optimalMismatchCount z z' ≤ permutationMismatchCount z z' σ := by
      unfold optimalMismatchCount
      apply Finset.min'_le
      exact Finset.mem_image.mpr ⟨σ, Finset.mem_univ σ, rfl⟩
    have hZero : permutationMismatchCount z z' σ = 0 := by
      simp [permutationMismatchCount, hσ]
    exact Nat.eq_zero_of_le_zero (hZero ▸ hLe)
  · intro hZero
    have hMem :
        optimalMismatchCount z z' ∈
          Finset.univ.image (permutationMismatchCount z z') := by
      unfold optimalMismatchCount
      apply Finset.min'_mem
    rw [hZero] at hMem
    rcases Finset.mem_image.mp hMem with ⟨σ, -, hσ⟩
    refine ⟨σ, ?_⟩
    apply hammingDist_eq_zero.mp
    simpa [permutationMismatchCount] using hσ

/--
Informal `d_Z(z,z')` ↔ Lean: minimum global-permutation Hamming error divided by the number of
centers.  The definition is total, but results that recover the numerator from this real quotient
must expose that the center type is nonempty.
-/
noncomputable def labelDistance {Center : Type u} {Label : Type v}
    [Fintype Center] [DecidableEq Center] [Fintype Label] [DecidableEq Label]
    (z z' : Partition Center Label) : ℝ :=
  (optimalMismatchCount z z' : ℝ) / Fintype.card Center

/-- Informal exact recovery ↔ Lean: equality of label orbits, not coordinatewise label equality. -/
def Exact {Center : Type u} {Label : Type v}
    (candidate truth : Partition Center Label) : Prop :=
  LabelEquivalent candidate truth

/-- Zero permutation-invariant label distance is exactly equality modulo global label switching. -/
theorem exact_iff_labelDistance_eq_zero
    {Center : Type u} {Label : Type v}
    [Fintype Center] [DecidableEq Center] [Nonempty Center]
    [Fintype Label] [DecidableEq Label]
    {candidate truth : Partition Center Label} :
    Exact candidate truth ↔ labelDistance candidate truth = 0 := by
  rw [Exact, labelDistance, div_eq_zero_iff]
  simp only [Nat.cast_eq_zero, Fintype.card_ne_zero, or_false]
  exact labelEquivalent_iff_optimalMismatchCount_eq_zero

end BiostatFormal.HighDimensional.Multiview
