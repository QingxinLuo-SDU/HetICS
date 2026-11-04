import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

/-! First-batch deterministic interfaces for the binary pipeline.
No label recovery, nuisance fitting, causal identification, or limit theorem is asserted. -/

open scoped BigOperators

namespace BinaryCore

variable {I K : Type*} [DecidableEq I] [DecidableEq K]

/-- Full-sample reuse with an internal, disjoint A/B split. -/
structure FullSample (I : Type*) [DecidableEq I] where
  S : Finset I
  A : Finset I
  B : Finset I
  label : Finset I
  screen : Finset I
  effect : Finset I
  split_disjoint : Disjoint A B
  split_cover : A ∪ B = S
  label_full : label = S
  screen_full : screen = S
  effect_full : effect = S

theorem direction_exactly_one (d : FullSample I) {i : I} (hi : i ∈ d.S) :
    (i ∈ d.A ∧ i ∉ d.B) ∨ (i ∈ d.B ∧ i ∉ d.A) := by
  have hc : i ∈ d.A ∨ i ∈ d.B := by
    simpa only [← d.split_cover, Finset.mem_union] using hi
  have hd := Finset.disjoint_left.mp d.split_disjoint
  rcases hc with ha | hb
  · exact Or.inl ⟨ha, fun hb => hd ha hb⟩
  · exact Or.inr ⟨hb, fun ha => hd ha hb⟩

theorem direction_counts (d : FullSample I) : d.A.card + d.B.card = d.S.card := by
  rw [← d.split_cover, Finset.card_union_of_disjoint d.split_disjoint]

theorem all_stages_use_full_sample (d : FullSample I) (i : I) :
    (i ∈ d.label ↔ i ∈ d.S) ∧ (i ∈ d.screen ↔ i ∈ d.S) ∧
      (i ∈ d.effect ↔ i ∈ d.S) := by
  simp [d.label_full, d.screen_full, d.effect_full]

/-- A total label map gives disjoint fibers. Every reported cluster and S are nonempty. -/
structure SamplePartition (I K : Type*) [DecidableEq I] [DecidableEq K] where
  S : Finset I
  cluster : I → K
  sample_nonempty : S.Nonempty
  fiber_nonempty : ∀ k, (S.filter (fun i => cluster i = k)).Nonempty

namespace SamplePartition

variable (p : SamplePartition I K)

def fiber (k : K) : Finset I := p.S.filter (fun i => p.cluster i = k)
def n (k : K) : ℕ := (p.fiber k).card
def N : ℕ := p.S.card
noncomputable def weight (k : K) : ℝ := (p.n k : ℝ) / p.N
noncomputable def clusterMean (score : I → ℝ) (k : K) : ℝ :=
  (∑ i ∈ p.fiber k, score i) / p.n k
noncomputable def fullMean (score : I → ℝ) : ℝ := (∑ i ∈ p.S, score i) / p.N

theorem fiber_cover (i : I) : (∃ k, i ∈ p.fiber k) ↔ i ∈ p.S := by
  simp [fiber]

theorem fiber_disjoint {k l : K} (hkl : k ≠ l) : Disjoint (p.fiber k) (p.fiber l) := by
  apply Finset.disjoint_left.mpr
  intro i hi hj
  exact hkl ((Finset.mem_filter.mp hi).2.symm.trans (Finset.mem_filter.mp hj).2)

theorem counts_decompose [Fintype K] : ∑ k, p.n k = p.N := by
  simpa [n, N, fiber] using
    (Finset.card_eq_sum_card_fiberwise (s := p.S) (t := Finset.univ)
      (f := p.cluster) (fun _ _ => Finset.mem_univ _)).symm

theorem scores_decompose [Fintype K] (score : I → ℝ) :
    ∑ k, ∑ i ∈ p.fiber k, score i = ∑ i ∈ p.S, score i := by
  classical
  simpa [fiber] using
    (Finset.sum_fiberwise_of_maps_to (s := p.S) (t := Finset.univ)
      (g := p.cluster) (f := score) (fun _ _ => Finset.mem_univ _))

theorem n_pos (k : K) : 0 < p.n k := Finset.card_pos.mpr (p.fiber_nonempty k)
theorem N_pos : 0 < p.N := Finset.card_pos.mpr p.sample_nonempty

theorem weight_pos (k : K) : 0 < p.weight k := by
  unfold weight
  exact div_pos (by exact_mod_cast p.n_pos k) (by exact_mod_cast p.N_pos)

theorem weights_sum_one [Fintype K] : ∑ k, p.weight k = 1 := by
  unfold weight
  rw [← Finset.sum_div, ← Nat.cast_sum, p.counts_decompose]
  exact div_self (by exact_mod_cast (ne_of_gt p.N_pos))

theorem weighted_mean_eq_full [Fintype K] (score : I → ℝ) :
    ∑ k, p.weight k * p.clusterMean score k = p.fullMean score := by
  have hn (k : K) : (p.n k : ℝ) ≠ 0 := by exact_mod_cast (ne_of_gt (p.n_pos k))
  have hN : (p.N : ℝ) ≠ 0 := by exact_mod_cast (ne_of_gt p.N_pos)
  calc
    _ = ∑ k, (∑ i ∈ p.fiber k, score i) / (p.N : ℝ) := by
      apply Finset.sum_congr rfl
      intro k _
      unfold weight clusterMean
      field_simp [hn k, hN]
    _ = p.fullMean score := by
      rw [← Finset.sum_div, p.scores_decompose]
      rfl

end SamplePartition

/-- The actual row score, with treated residual minus control residual. -/
noncomputable def phi (x y e m0 m1 : ℝ) : ℝ :=
  m1 - m0 + x / e * (y - m1) - (1 - x) / (1 - e) * (y - m0)

theorem weighted_aipw_eq_full [Fintype K] (p : SamplePartition I K)
    (x y e m0 m1 : I → ℝ) :
    (∑ k, p.weight k * p.clusterMean (fun i => phi (x i) (y i) (e i) (m0 i) (m1 i)) k) =
      p.fullMean (fun i => phi (x i) (y i) (e i) (m0 i) (m1 i)) :=
  p.weighted_mean_eq_full _

/-- Parameters of a four-state binary law, fixed at the conditioning value in question. -/
structure BinaryLaw where
  e0 : ℝ
  mu0 : ℝ
  mu1 : ℝ
  e0_mem : 0 ≤ e0 ∧ e0 ≤ 1
  mu0_mem : 0 ≤ mu0 ∧ mu0 ≤ 1
  mu1_mem : 0 ≤ mu1 ∧ mu1 ≤ 1

namespace BinaryLaw

noncomputable def p11 (b : BinaryLaw) : ℝ := b.e0 * b.mu1
noncomputable def p10 (b : BinaryLaw) : ℝ := b.e0 * (1 - b.mu1)
noncomputable def p01 (b : BinaryLaw) : ℝ := (1 - b.e0) * b.mu0
noncomputable def p00 (b : BinaryLaw) : ℝ := (1 - b.e0) * (1 - b.mu0)

theorem masses_probability (b : BinaryLaw) :
    (0 ≤ b.p11 ∧ 0 ≤ b.p10 ∧ 0 ≤ b.p01 ∧ 0 ≤ b.p00) ∧
      b.p11 + b.p10 + b.p01 + b.p00 = 1 := by
  constructor
  · exact ⟨mul_nonneg b.e0_mem.1 b.mu1_mem.1,
      mul_nonneg b.e0_mem.1 (sub_nonneg.mpr b.mu1_mem.2),
      mul_nonneg (sub_nonneg.mpr b.e0_mem.2) b.mu0_mem.1,
      mul_nonneg (sub_nonneg.mpr b.e0_mem.2) (sub_nonneg.mpr b.mu0_mem.2)⟩
  · unfold p11 p10 p01 p00
    ring

noncomputable def fourStateMean (b : BinaryLaw) (e m0 m1 : ℝ) : ℝ :=
  b.p11 * phi 1 1 e m0 m1 + b.p10 * phi 1 0 e m0 m1 +
    b.p01 * phi 0 1 e m0 m1 + b.p00 * phi 0 0 e m0 m1

/-- Algebraic form of the old aipwConditionalMean; no old causal model is imported. -/
noncomputable def conditionalMean (b : BinaryLaw) (e m0 m1 : ℝ) : ℝ :=
  m1 - m0 + b.e0 / e * (b.mu1 - m1) - (1 - b.e0) / (1 - e) * (b.mu0 - m0)

theorem four_state_eq_conditional (b : BinaryLaw) (e m0 m1 : ℝ) :
    b.fourStateMean e m0 m1 = b.conditionalMean e m0 m1 := by
  unfold fourStateMean conditionalMean p11 p10 p01 p00 phi
  simp only [zero_div, zero_mul, sub_self, sub_zero]
  ring

theorem four_state_bias (b : BinaryLaw) (e m0 m1 : ℝ) (he : 0 < e) (he1 : e < 1) :
    b.fourStateMean e m0 m1 = b.mu1 - b.mu0 +
      (b.e0 - e) * ((b.mu1 - m1) / e + (b.mu0 - m0) / (1 - e)) := by
  rw [b.four_state_eq_conditional]
  unfold conditionalMean
  have h0 : e ≠ 0 := ne_of_gt he
  have h1 : 1 - e ≠ 0 := ne_of_gt (sub_pos.mpr he1)
  field_simp [h0, h1]
  ring

theorem propensity_correct (b : BinaryLaw) (m0 m1 : ℝ)
    (he : 0 < b.e0) (he1 : b.e0 < 1) :
    b.fourStateMean b.e0 m0 m1 = b.mu1 - b.mu0 := by
  rw [b.four_state_bias b.e0 m0 m1 he he1]
  ring

theorem outcome_correct (b : BinaryLaw) (e : ℝ) (he : 0 < e) (he1 : e < 1) :
    b.fourStateMean e b.mu0 b.mu1 = b.mu1 - b.mu0 := by
  rw [b.four_state_bias e b.mu0 b.mu1 he he1]
  simp

end BinaryLaw

#print axioms direction_exactly_one
#print axioms direction_counts
#print axioms all_stages_use_full_sample
#print axioms SamplePartition.fiber_cover
#print axioms SamplePartition.fiber_disjoint
#print axioms SamplePartition.counts_decompose
#print axioms SamplePartition.scores_decompose
#print axioms SamplePartition.weight_pos
#print axioms SamplePartition.weights_sum_one
#print axioms SamplePartition.weighted_mean_eq_full
#print axioms weighted_aipw_eq_full
#print axioms BinaryLaw.masses_probability
#print axioms BinaryLaw.four_state_eq_conditional
#print axioms BinaryLaw.four_state_bias
#print axioms BinaryLaw.propensity_correct
#print axioms BinaryLaw.outcome_correct

end BinaryCore
