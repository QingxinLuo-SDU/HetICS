import FixedCenterMConsumer
import LearnedRuleApplicationConsumer
import HierarchicalEffectsConsumer
import HierarchicalPopulationConsumer
import RegisteredCoreBridge
import CenterClusterInferenceConsumer

noncomputable section
open Filter MeasureTheory ProbabilityTheory Matrix FixedCenterM
open scoped Topology BigOperators Matrix.Norms.Elementwise
namespace BinarySubmission

/-- The coordinate operator is constructed from the very matrix used in the sandwich. -/
def matrixMap {D : Type*} [Fintype D] (M : Matrix D D ℝ) : (D → ℝ) →L[ℝ] (D → ℝ) :=
  ⟨M.mulVecLin, by
    change Continuous (fun x => M *ᵥ x)
    unfold Matrix.mulVec dotProduct
    fun_prop⟩

theorem matrixMap_inverse {D : Type*} [Fintype D] [DecidableEq D]
    (M : Matrix D D ℝ) (hu : IsUnit M.det) (x : D → ℝ) :
    matrixMap M⁻¹ (matrixMap M x) = x := by
  change M⁻¹ *ᵥ (M *ᵥ x) = x
  rw [Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul M hu, Matrix.one_mulVec]

theorem target_row_bound {D : Type*} [Fintype D] (h x : D → ℝ)
    (hp : ∀ j, 0 ≤ h j) (hs : ∑ j, h j = 1) : |h ⬝ᵥ x| ≤ ‖x‖ := by
  calc
    |h ⬝ᵥ x| ≤ ∑ j, |h j * x j| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ j, h j * |x j| := by simp_rw [abs_mul, abs_of_nonneg (hp _)]
    _ ≤ ∑ j, h j * ‖x‖ := Finset.sum_le_sum fun j _ =>
      mul_le_mul_of_nonneg_left (by simpa using norm_le_pi_norm x j) (hp j)
    _ = ‖x‖ := by rw [← Finset.sum_mul, hs, one_mul]

/-- Instantiate the extraction bound with actual nonempty-fiber person counts. -/
theorem person_target_bound {I K : Type*} [DecidableEq I] [DecidableEq K]
    [Fintype K] (p : BinaryCore.SamplePartition I K) (x : K → ℝ) :
    |p.weight ⬝ᵥ x| ≤ ‖x‖ :=
  target_row_bound p.weight x (fun k => le_of_lt (p.weight_pos k)) p.weights_sum_one

section Reference
variable {Ω C D : Type*} [MeasurableSpace Ω] [Fintype C] [Fintype D] [DecidableEq D]
  {I : ℕ → C → Type*} [∀ n c, Fintype (I n c)] [∀ n c, Nonempty (I n c)]

/-- Data, not limit conclusions. For applications the row map is the zero-padded
actual logistic/AIPW joint equation; the same map defines reference and fitted rows.
Finite n below the physical sample range are proof extensions only. -/
structure ReferenceData (Ω C D : Type*) (I : ℕ → C → Type*) [Fintype D] where
  score : (n : ℕ) → Ω → (c : C) → I n c → (D → ℝ) → D → ℝ
  eta : ℕ → D → ℝ
  etaHat : ℕ → Ω → D → ℝ
  w : ℕ → Ω → C → ℝ
  h : ℕ → D → ℝ
  Aref : ℕ → Ω → Matrix D D ℝ
  Apert : ℕ → Ω → Matrix D D ℝ
  A0 : Matrix D D ℝ
  h0 : D → ℝ
  pi : C → ℝ
  m : C → D → ℝ
  Q : C → Matrix D D ℝ
  T : ℕ → Ω → (D → ℝ) →L[ℝ] (D → ℝ)
  xi : (n : ℕ) → Ω → Fin (n+1) → ℝ

def ReferenceData.u (b : ReferenceData Ω C D I) (n : ℕ) (ω : Ω) (c : C) (i : I n c) :=
  b.score n ω c i (b.eta n)
def ReferenceData.v (b : ReferenceData Ω C D I) (n : ℕ) (ω : Ω) (c : C) (i : I n c) :=
  b.score n ω c i (b.etaHat n ω)
def ReferenceData.s (b : ReferenceData Ω C D I) (n : ℕ) (ω : Ω) : D → ℝ :=
  fun j => Real.sqrt ((n:ℝ)+1) * ∑ c, b.w n ω c * avg (fun i => b.u n ω c i j)
def ReferenceData.r (b : ReferenceData Ω C D I) (n : ℕ) (ω : Ω) : D → ℝ :=
  fun j => Real.sqrt ((n:ℝ)+1) * ∑ c, b.w n ω c * avg (fun i => b.v n ω c i j)
def ReferenceData.delta (b : ReferenceData Ω C D I) (n : ℕ) (ω : Ω) : D → ℝ :=
  fun j => Real.sqrt ((n:ℝ)+1) * (b.etaHat n ω j-b.eta n j)
def ReferenceData.point (b : ReferenceData Ω C D I) (n : ℕ) (ω : Ω) := b.h n ⬝ᵥ b.etaHat n ω
def ReferenceData.target (b : ReferenceData Ω C D I) (n : ℕ) := b.h n ⬝ᵥ b.eta n
def ReferenceData.rM (b : ReferenceData Ω C D I) (n : ℕ) (ω : Ω) :=
  b.point n ω-b.target n-ErrorTarget.linearRowMean b.xi n ω
def ReferenceData.scaledVariance (b : ReferenceData Ω C D I) (n : ℕ) (ω : Ω) :=
  sandwichScalar (b.h n) ((b.Aref n ω+b.Apert n ω)⁻¹) (actualMeat (b.w n ω) (b.v n ω))
def ReferenceData.variance (b : ReferenceData Ω C D I) (n : ℕ) (ω : Ω) :=
  b.scaledVariance n ω / ((n:ℝ)+1)
def ReferenceData.sigmaSq (b : ReferenceData Ω C D I) :=
  sandwichScalar b.h0 b.A0⁻¹ (meatFromMoments b.pi b.m b.Q)

/-- Actual coordinate derivative of the same row function, along an additive
coordinate path. This is a deterministic application interface. -/
def ReferenceData.derivativeAt (b : ReferenceData Ω C D I) (n : ℕ) (ω : Ω)
    (c : C) (i : I n c) (eta : D → ℝ) (j k : D) : ℝ :=
  deriv (fun t : ℝ => b.score n ω c i (eta + t • Pi.single k 1) j) 0

/-- Primitive moment limits and quantitative derivative/score/residual bounds.
The row identity is a deterministic coordinate bridge, not an estimator CLT. -/
structure ReferenceConditions (μ : Measure Ω) (b : ReferenceData Ω C D I) where
  unit : IsUnit b.A0.det
  bread_coordinates : ∀ n ω j k, (b.Aref n ω+b.Apert n ω) j k =
    -(∑ c, b.w n ω c * avg (fun i => b.derivativeAt n ω c i (b.etaHat n ω) j k))
  positive : ∀ n j, 0 ≤ b.h n j
  sum_one : ∀ n, ∑ j, b.h n j = 1
  weights : ∀ c, TendstoInMeasure μ (fun (n : ℕ) ω => b.w n ω c) atTop (fun _ => b.pi c)
  first : ∀ c j, TendstoInMeasure μ (fun (n : ℕ) ω => avg (fun i => b.u n ω c i j)) atTop (fun _ => b.m c j)
  second : ∀ c j k, TendstoInMeasure μ (fun (n : ℕ) ω => avg (fun i => b.u n ω c i j*b.u n ω c i k)) atTop (fun _ => b.Q c j k)
  perturb : ∀ c j, TendstoInMeasure μ (fun (n : ℕ) ω => avg (fun i => (b.v n ω c i j-b.u n ω c i j)^2)) atTop 0
  row_limit : TendstoInMeasure μ (fun n _ => b.h n) atTop (fun _ => b.h0)
  bread : TendstoInMeasure μ b.Aref atTop (fun _ => b.A0)
  bread_perturb : TendstoInMeasure μ b.Apert atTop 0
  d : ℕ → ℝ
  z : ℕ → ℝ
  q : ℕ → ℝ
  good : ℕ → Set Ω
  segment : ∀ (n : ℕ) ω, ω ∈ good n → b.T n ω (b.delta n ω) = b.s n ω-b.r n ω
  bounds : ∀ (n : ℕ) ω, ω ∈ good n → ‖matrixMap b.A0-b.T n ω‖ ≤ d n ∧ ‖b.s n ω‖ ≤ z n ∧ ‖b.r n ω‖ ≤ q n
  nonneg : ∀ n, 0 ≤ d n ∧ 0 ≤ z n ∧ 0 ≤ q n
  small : ∀ n, ‖matrixMap b.A0⁻¹‖*d n ≤ 1/2
  dz : Tendsto (fun n => d n*z n) atTop (𝓝 0)
  dq : Tendsto (fun n => d n*q n) atTop (𝓝 0)
  q_zero : Tendsto q atTop (𝓝 0)
  good_prob : Tendsto (fun n => μ (good n)ᶜ) atTop (𝓝 0)
  centered_rows : ∀ (n : ℕ) ω, Real.sqrt ((n:ℝ)+1)*ErrorTarget.linearRowMean b.xi n ω =
    b.h n ⬝ᵥ (b.A0⁻¹ *ᵥ b.s n ω)

omit [MeasurableSpace Ω] [∀ n c, Nonempty (I n c)] in
theorem reference_variance_scale (b : ReferenceData Ω C D I) (n : ℕ) (ω : Ω) :
    ((n:ℝ)+1)*b.variance n ω = b.scaledVariance n ω := by
  unfold ReferenceData.variance
  field_simp

theorem reference_raw_variance (μ : Measure Ω) (b : ReferenceData Ω C D I)
    (hc : ReferenceConditions μ b) :
    TendstoInMeasure μ (fun (n : ℕ) ω => ((n:ℝ)+1)*b.variance n ω) atTop (fun _ => b.sigmaSq) := by
  simp_rw [reference_variance_scale]
  exact FixedCenterMConsumer.actual_reference_variance μ b.w b.u b.v b.pi b.m b.Q
    (fun n _ => b.h n) b.Aref b.Apert b.h0 b.A0 hc.unit hc.weights hc.first hc.second
    hc.perturb hc.row_limit hc.bread hc.bread_perturb

omit [∀ n c, Nonempty (I n c)] in
theorem reference_target_remainder_identity {μ : Measure Ω} (b : ReferenceData Ω C D I)
    (hc : ReferenceConditions μ b) (n : ℕ) (ω : Ω) :
    Real.sqrt ((n:ℝ)+1)*b.rM n ω =
      b.h n ⬝ᵥ (b.delta n ω-matrixMap b.A0⁻¹ (b.s n ω)) := by
  rw [dotProduct_sub]
  change _ = b.h n ⬝ᵥ b.delta n ω-b.h n ⬝ᵥ (b.A0⁻¹ *ᵥ b.s n ω)
  rw [← hc.centered_rows]
  unfold ReferenceData.rM ReferenceData.point ReferenceData.target ReferenceData.delta dotProduct
  simp_rw [mul_sub, Finset.sum_sub_distrib]
  rw [Finset.mul_sum, Finset.mul_sum]
  simp only [mul_left_comm]

omit [∀ n c, Nonempty (I n c)] in
theorem reference_raw_remainder (μ : Measure Ω) (b : ReferenceData Ω C D I)
    (hc : ReferenceConditions μ b) (σ : ℝ) :
    TendstoInMeasure μ (fun (n : ℕ) ω => Real.sqrt ((n:ℝ)+1)*b.rM n ω/σ) atTop 0 := by
  have hv := linear_remainder_inMeasure μ (matrixMap b.A0) (matrixMap b.A0⁻¹)
    b.T b.delta b.s b.r hc.d hc.z hc.q hc.good (matrixMap_inverse b.A0 hc.unit)
    hc.segment hc.bounds hc.nonneg hc.small hc.dz hc.dq hc.q_zero hc.good_prob
  have hs : TendstoInMeasure μ (fun (n : ℕ) ω => Real.sqrt ((n:ℝ)+1)*b.rM n ω) atTop 0 := by
    rw [tendstoInMeasure_iff_norm] at hv ⊢
    intro ε hε
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds (hv ε hε)
    · intro n; exact bot_le
    · intro n
      apply measure_mono
      intro ω hw
      simp only [Set.mem_setOf_eq, Pi.zero_apply, sub_zero, Real.norm_eq_abs] at hw ⊢
      rw [abs_of_nonneg (norm_nonneg _)]
      rw [reference_target_remainder_identity b hc] at hw
      exact hw.trans (target_row_bound _ _ (hc.positive n) (hc.sum_one n))
  have hd := inMeasure_continuousAt μ _ (0:ℝ) (fun x : ℝ => x/σ) hs (by fun_prop)
  simpa [tendstoInMeasure_iff_norm] using hd
end Reference

#print axioms matrixMap
#print axioms matrixMap_inverse
#print axioms target_row_bound
#print axioms person_target_bound
#print axioms reference_variance_scale
#print axioms reference_raw_variance
#print axioms reference_target_remainder_identity
#print axioms reference_raw_remainder
open BinaryFormal LearnedRuleApplication FiniteFreeze
open BiostatFormal.HighDimensional.Multiview

section Learning
variable {Ω Context Center Label Slot RC EC V Parameter : Type*}
  [Fintype Center] [DecidableEq Center] [Fintype Label] [DecidableEq Label]
  [DecidableEq (CandidateRecord Center Label Slot RC)] [DecidableEq V]

structure LearningData (Ω Context Center Label Slot RC EC V Parameter : Type*)
    [DecidableEq V] where
  ordered : ℕ → Ω → List (Option (CandidateRecord Center Label Slot RC))
  spec : ℕ → SelectorRiskSpec (CandidateRecord Center Label Slot RC) EC
  truth : Partition Center Label
  u : ℕ → ℝ
  pipe : Pipeline Context Center V Parameter
  ctxA : ℕ → Ω → Context
  ctxR : ℕ → Ω → Context
  p : ℕ → Ω → Parameter
  S0 : ℕ → Center → Finset V
  refit : ℕ → Ω → Option (Refit Center Label Parameter)
  fallbackA : ℕ → Ω → Report
  fallbackC : ℕ → Ω → Report

def LearningData.reportA (a : LearningData Ω Context Center Label Slot RC EC V Parameter)
    (n : ℕ) (ω : Ω) := actual a.pipe (a.ctxA n ω)
      (B1Current.frozenOutput (a.ordered n ω) (a.spec n) (a.refit n ω))
def LearningData.reportC (a : LearningData Ω Context Center Label Slot RC EC V Parameter)
    (n : ℕ) (ω : Ω) := reference a.pipe (a.ctxR n ω) (a.p n ω) a.truth (a.S0 n)
def LearningData.pointA (a : LearningData Ω Context Center Label Slot RC EC V Parameter)
    (n : ℕ) (ω : Ω) := (extend (a.fallbackA n ω) (a.reportA n ω)).point
def LearningData.varianceA (a : LearningData Ω Context Center Label Slot RC EC V Parameter)
    (n : ℕ) (ω : Ω) := (extend (a.fallbackA n ω) (a.reportA n ω)).variance

variable [MeasurableSpace Ω]
structure LearningConditions (μ : Measure Ω)
    (a : LearningData Ω Context Center Label Slot RC EC V Parameter) where
  L : ℕ → Set Ω
  S : ℕ → Set Ω
  U : ℕ → Set Ω
  δL : ℕ → ENNReal
  δS : ℕ → ENNReal
  δU : ℕ → ENNReal
  b1 : ∀ (n : ℕ) ω, ω ∈ L n → ω ∈
    exactSurvivalEvent (B1Current.currentPool (a.ordered n)) a.truth ∩
    uniformDeviationEvent (B1Current.currentPool (a.ordered n)) (a.spec n) (a.u n) ∩
    selectorStrictPopulationGapEvent (B1Current.currentPool (a.ordered n)) (a.spec n) a.truth (a.u n) 0 0 0
  stable : ∀ (n : ℕ) ω, ω ∈ S n → stableEvent a.pipe (a.ctxR n ω) a.truth (a.S0 n)
  usable : ∀ (n : ℕ) ω, ω ∈ U n → usableEvent a.pipe (a.ctxA n ω) (a.ctxR n ω) (a.p n ω)
    a.truth (a.S0 n) (B1Current.frozenOutput (a.ordered n ω) (a.spec n) (a.refit n ω))
  bounds : ∀ n, μ (L n)ᶜ ≤ δL n ∧ μ (S n)ᶜ ≤ δS n ∧ μ (U n)ᶜ ≤ δU n
  label_zero : Tendsto δL atTop (𝓝 0)
  stable_zero : Tendsto δS atTop (𝓝 0)
  usable_zero : Tendsto δU atTop (𝓝 0)

/-- The fixed-reference report fields are bound to the exact reference data above.
Thus the moment, Taylor, B1 and reporting branches share their point and variance. -/
theorem submission_studentized {D : Type*} [Fintype D] [DecidableEq D]
    {I : ℕ → Center → Type*} [∀ n c, Fintype (I n c)] [∀ n c, Nonempty (I n c)]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (b : ReferenceData Ω Center D I) (hc : ReferenceConditions μ b)
    (a : LearningData Ω Context Center Label Slot RC EC V Parameter) (hl : LearningConditions μ a)
    (point_match : ∀ (n : ℕ) ω, (extend (a.fallbackC n ω) (a.reportC n ω)).point = b.point n ω)
    (variance_match : ∀ (n : ℕ) ω, (extend (a.fallbackC n ω) (a.reportC n ω)).variance = b.variance n ω)
    (σ : ℝ) (hσ : 0 < σ) (sigma_match : b.sigmaSq = σ^2)
    (rowCLT : TendstoInDistribution
      (fun (n : ℕ) ω => Real.sqrt ((n:ℝ)+1)*ErrorTarget.linearRowMean b.xi n ω/σ)
      atTop id (fun _ => μ) (gaussianReal 0 1))
    (hmM : ∀ n, AEMeasurable (b.rM n) μ)
    (hmSel : ∀ n, AEMeasurable (fun ω => a.pointA n ω-b.point n ω) μ)
    (hmV : ∀ n, AEMeasurable (a.varianceA n) μ) :
    TendstoInDistribution (fun (n : ℕ) ω => (a.pointA n ω-b.target n)/Real.sqrt (a.varianceA n ω))
      atTop id (fun _ => μ) (gaussianReal 0 1) ∧
    TendstoInMeasure μ (fun (n : ℕ) ω => ((n:ℝ)+1)*a.varianceA n ω) atTop (fun _ => σ^2) ∧
    Tendsto (fun n => μ {ω | a.reportA n ω = none}) atTop (𝓝 0) := by
  have ht := LearnedRuleApplicationConsumer.b1_to_two_scales μ a.ordered a.spec a.truth a.u
    a.pipe a.ctxA a.ctxR a.p a.S0 a.refit hl.L hl.S hl.U hl.δL hl.δS hl.δU
    hl.b1 hl.stable hl.usable hl.bounds hl.label_zero hl.stable_zero hl.usable_zero a.fallbackA a.fallbackC
  change TendstoInMeasure μ (fun (n : ℕ) ω => Real.sqrt ((n:ℝ)+1)*
      (a.pointA n ω-(extend (a.fallbackC n ω) (a.reportC n ω)).point)) atTop 0 ∧
    TendstoInMeasure μ (fun (n : ℕ) ω => ((n:ℝ)+1)*
      (a.varianceA n ω-(extend (a.fallbackC n ω) (a.reportC n ω)).variance)) atTop 0 ∧
    Tendsto (fun n => μ {ω | a.reportA n ω = none}) atTop (𝓝 0) at ht
  simp_rw [point_match, variance_match] at ht
  have hsel := inMeasure_continuousAt μ _ (0:ℝ) (fun x : ℝ => x/σ) ht.1 (by fun_prop)
  have hv := reference_raw_variance μ b hc
  rw [sigma_match] at hv
  have hm := reference_raw_remainder μ b hc σ
  have hstud := ErrorTarget.actual_ate_studentized μ a.pointA b.target b.xi b.rM
    (fun (n : ℕ) ω => a.pointA n ω-b.point n ω) b.variance a.varianceA σ hσ
    (fun (n : ℕ) ω => by unfold ReferenceData.rM; ring) rowCLT hm (by simpa [tendstoInMeasure_iff_norm] using hsel)
    hmM hmSel hv ht.2.1 hmV
  refine ⟨hstud, ?_, ht.2.2⟩
  exact ErrorTarget.variance_consistency_transfer μ _ _ (σ^2) hv (by simpa only [mul_sub] using ht.2.1)
end Learning

/-- Consume the three outputs of submission_studentized. Unavailable and nonpositive
variance reports stay in the denominator and count as noncoverage. -/
theorem submission_reported_coverage {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (report : ℕ → Ω → Option Report)
    (point variance : ℕ → Ω → ℝ) (target : ℕ → ℝ) (σ z : ℝ) (hσ : 0 < σ) (hz : 0 ≤ z)
    (hT : TendstoInDistribution (fun (n : ℕ) ω => (point n ω-target n)/Real.sqrt (variance n ω))
      atTop id (fun _ => μ) (gaussianReal 0 1))
    (hV : TendstoInMeasure μ (fun (n : ℕ) ω => ((n:ℝ)+1)*variance n ω) atTop (fun _ => σ^2))
    (hfail : Tendsto (fun n => μ {ω | report n ω = none}) atTop (𝓝 0))
    (hmeas : ∀ n, MeasurableSet {ω | report n ω ≠ none ∧ 0 < variance n ω}) :
    Tendsto (fun n => μ.real {ω | (report n ω ≠ none ∧ 0 < variance n ω) ∧
      point n ω-z*Real.sqrt (variance n ω) ≤ target n ∧
      target n ≤ point n ω+z*Real.sqrt (variance n ω)}) atTop
      (𝓝 ((gaussianReal 0 1).real (Set.Icc (-z) z))) := by
  have hvbad := ErrorTarget.nonpositive_variance_probability μ variance σ hσ hV
  have hav : Tendsto (fun n => μ {ω | report n ω ≠ none ∧ 0 < variance n ω}ᶜ) atTop (𝓝 0) := by
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds (by simpa using hfail.add hvbad)
    · intro n; exact bot_le
    · intro n
      apply le_trans (measure_mono ?_) (measure_union_le _ _)
      intro ω hw
      simp only [Set.mem_compl_iff, Set.mem_setOf_eq, not_and_or, not_not, not_lt] at hw
      exact hw
  exact ErrorTarget.reported_interval_coverage μ point (fun (n : ℕ) ω => Real.sqrt (variance n ω))
    target z hz _ hmeas (fun (n : ℕ) ω hw => Real.sqrt_pos.2 hw.2) hav hT

#print axioms submission_studentized
#print axioms submission_reported_coverage

/-- A full-coordinate logistic/AIPW row function for instantiating ReferenceData.score. -/
def aipwCoordinates {p : Type*} [Fintype p] (design : p → ℝ) (x y : ℝ)
    (eta : (Fin 3 × p) ⊕ Unit → ℝ) : (Fin 3 × p) ⊕ Unit → ℝ :=
  AIPWJacobian.joint design x y (fun a j => eta (.inl (a,j))) (eta (.inr ()))

theorem aipw_coordinate_path {p : Type*} [Fintype p] [DecidableEq p]
    (design : p → ℝ) (x y : ℝ) (eta : (Fin 3 × p) ⊕ Unit → ℝ)
    (k : (Fin 3 × p) ⊕ Unit) (t : ℝ) :
    aipwCoordinates design x y (eta + t • Pi.single k 1) =
      AIPWJacobian.jointPath design x y (fun a j => eta (.inl (a,j))) (eta (.inr ())) k t := by
  cases k with
  | inl k =>
    unfold aipwCoordinates AIPWJacobian.jointPath
    congr 1
    · funext a j
      simp [AIPWJacobian.move, Pi.single_apply, Prod.ext_iff]
    · simp
  | inr k =>
    cases k
    unfold aipwCoordinates AIPWJacobian.jointPath
    congr 1
    · funext a j; simp
    · simp

/-- Negative average coordinate derivatives of aipwCoordinates are exactly the
old actualBread; this instantiates ReferenceConditions.bread_coordinates. -/
theorem aipw_coordinate_bread {p I : Type*} [Fintype p] [DecidableEq p]
    [Fintype I] (design : I → p → ℝ) (x y : I → ℝ)
    (eta : (Fin 3 × p) ⊕ Unit → ℝ) (j k : (Fin 3 × p) ⊕ Unit) :
    -(avg (fun i => deriv (fun t : ℝ => aipwCoordinates (design i) (x i) (y i)
      (eta + t • Pi.single k 1) j) 0)) =
    AIPWJacobian.actualBread design x y (fun a j => eta (.inl (a,j))) (eta (.inr ())) j k := by
  simp_rw [aipw_coordinate_path]
  rfl

/-- Concrete actual score, actual derivative bread, centered moments and the SAME
matrix inverse's continuous linear action. This supplies the local AIPW instance. -/
theorem aipw_actual_instance {p I : Type*} [Fintype p] [DecidableEq p]
    [Fintype I] [Nonempty I] (design : I → p → ℝ) (x y : I → ℝ)
    (eta : (Fin 3 × p) ⊕ Unit → ℝ)
    (hu : IsUnit (AIPWJacobian.actualBread design x y
      (fun a j => eta (.inl (a,j))) (eta (.inr ()))).det)
    (j k : (Fin 3 × p) ⊕ Unit) (v : (Fin 3 × p) ⊕ Unit → ℝ) :
    let A := AIPWJacobian.actualBread design x y (fun a j => eta (.inl (a,j))) (eta (.inr ()))
    A = JointSandwich.bread (AIPWJacobian.meanInfo design x (fun a j => eta (.inl (a,j))))
      (AIPWJacobian.meanGrad design x y (fun a j => eta (.inl (a,j)))) ∧
    matrixMap A⁻¹ (matrixMap A v) = v ∧
    centeredMoment (fun i => aipwCoordinates (design i) (x i) (y i) eta) j k =
      avg (fun i => aipwCoordinates (design i) (x i) (y i) eta j *
        aipwCoordinates (design i) (x i) (y i) eta k) -
      avg (fun i => aipwCoordinates (design i) (x i) (y i) eta j) *
        avg (fun i => aipwCoordinates (design i) (x i) (y i) eta k) := by
  exact ⟨AIPWJacobian.actualBread_eq _ _ _ _ _, matrixMap_inverse _ hu v, centeredMoment_eq _ _ _⟩

/-- The general H extraction uses the actual SamplePartition and the complete
centered covariance. The displayed scalar is exactly V, not N times V. -/
theorem person_covariance_scale {I K C D : Type*} [DecidableEq I] [DecidableEq K]
    [Fintype K] [DecidableEq C] [Fintype D]
    (p : BinaryCore.SamplePartition I K) (center : I → C)
    (R : Matrix D D ℝ) (H : Matrix K D ℝ) (u : p.S → D → ℝ) :
    sandwichScalar (H.transpose *ᵥ p.weight) R
      (JointSandwich.meat (p.N:ℝ) (fun i => u i-JointSandwich.finiteMean
        (PersonWeightedJoint.centerRows p center i) u)) / (p.N:ℝ) =
    ((p.N:ℝ)^2)⁻¹ * ∑ i, (p.weight ⬝ᵥ (H *ᵥ (R *ᵥ u i)) -
      ((PersonWeightedJoint.centerRows p center i).card:ℝ)⁻¹ *
        ∑ j ∈ PersonWeightedJoint.centerRows p center i,
          p.weight ⬝ᵥ (H *ᵥ (R *ᵥ u j)))^2 := by
  have hs := PersonWeightedJoint.person_weighted_centered_sandwich p center R H u
  rw [← hs]
  simp only [← Matrix.mulVec_mulVec]
  rw [← Matrix.dotProduct_transpose_mulVec H, dotProduct_comm]
  simp [JointSandwich.covariance, sandwichScalar, Matrix.smul_mulVec,
    dotProduct_smul, div_eq_mul_inv, mul_comm]

/-- The fixed post-ICP sets used by the reference rule obtain the common causal
semantics from coverage+safety, and actual person counts aggregate those ATEs. -/
theorem qualified_person_target {X I K V : Type*} [MeasurableSpace X]
    [DecidableEq I] [DecidableEq K] [Fintype K] [DecidableEq V]
    (p : BinaryCore.SamplePartition I K) (μ : K → Measure X)
    (Y0 Y1 : K → X → ℝ) (hY0 : ∀ k, Integrable (Y0 k) (μ k))
    (hY1 : ∀ k, Integrable (Y1 k) (μ k))
    (core fixedS : K → Finset V) (safe : K → Finset V → Prop)
    (bridge : ∀ k, ScreeningAdjustment.CommonAdjustmentBridge (μ k) (Y0 k) (Y1 k) (core k) (safe k))
    (coverage : ∀ k, core k ⊆ fixedS k) (safety : ∀ k, safe k (fixedS k))
    (e m0 m1 : K → X → ℝ)
    (domain : ∀ k, ∀ᵐ x ∂μ k, 0 < e k x ∧ e k x < 1)
    (correct : ∀ k, (∀ᵐ x ∂μ k, e k x = ((bridge k).lawFor (fixedS k) x).e0) ∨
      (∀ᵐ x ∂μ k, m0 k x = ((bridge k).lawFor (fixedS k) x).mu0 ∧
        m1 k x = ((bridge k).lawFor (fixedS k) x).mu1)) :
    (∑ k, p.weight k * ScreeningAdjustment.populationAIPW (μ k)
      ((bridge k).lawFor (fixedS k)) (e k) (m0 k) (m1 k)) =
    ∑ k, p.weight k * ScreeningAdjustment.commonATE (μ k) (Y0 k) (Y1 k) := by
  exact ScreeningAdjustment.person_weighted_aipw_common_population p μ Y0 Y1 hY0 hY1 _
    (fun k => (bridge k).conditions_of_valid (fixedS k)
      (ScreeningAdjustment.final_valid_of_coverage_safe _ _ _ (coverage k) (safety k)))
    e m0 m1 domain correct

#print axioms aipwCoordinates
#print axioms aipw_coordinate_path
#print axioms aipw_coordinate_bread
#print axioms aipw_actual_instance
#print axioms person_covariance_scale
#print axioms qualified_person_target

/-- Center weights card(I_c)/N cancel the inner center denominator. Thus the
actual centered meat is the same once-normalized raw row sum used in JointSandwich. -/
theorem actualMeat_eq_jointMeat {C D : Type*} [Fintype C]
    {I : C → Type*} [∀ c, Fintype (I c)] [∀ c, Nonempty (I c)]
    (N : ℝ) (u : ∀ c, I c → D → ℝ) :
    actualMeat (fun c => (Fintype.card (I c):ℝ)/N) u =
      JointSandwich.meat N (fun i : Sigma I =>
        u i.1 i.2 - fun j => avg (fun l => u i.1 l j)) := by
  ext j k
  simp only [actualMeat, centeredMoment, avg, JointSandwich.meat,
    Matrix.smul_apply, Matrix.sum_apply, Matrix.vecMulVec_apply, Pi.sub_apply,
    smul_eq_mul, Fintype.sum_sigma]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro c _
  have hn : (Fintype.card (I c):ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  field_simp [hn]

#print axioms actualMeat_eq_jointMeat

/-- At a physical sample index, the very reference variance in the main theorem
is the previous joint covariance based on the same N=n+1 centered row sum. -/
theorem reference_variance_joint_rows {Ω C D : Type*} [Fintype C] [Fintype D] [DecidableEq D]
    {I : ℕ → C → Type*} [∀ n c, Fintype (I n c)] [∀ n c, Nonempty (I n c)]
    (b : ReferenceData Ω C D I) (n : ℕ) (ω : Ω)
    (hw : b.w n ω = fun c => (Fintype.card (I n c):ℝ)/((n:ℝ)+1)) :
    b.variance n ω = sandwichScalar (b.h n) ((b.Aref n ω+b.Apert n ω)⁻¹)
      (JointSandwich.meat ((n:ℝ)+1) (fun i : Sigma (I n) =>
        b.v n ω i.1 i.2 - fun j => avg (fun l => b.v n ω i.1 l j))) / ((n:ℝ)+1) := by
  unfold ReferenceData.variance ReferenceData.scaledVariance
  rw [hw, actualMeat_eq_jointMeat]

#print axioms reference_variance_joint_rows
section Complete
variable {Ω Context Center Label Slot RC EC V Parameter : Type*}
  [Fintype Center] [DecidableEq Center] [Fintype Label] [DecidableEq Label]
  [DecidableEq (CandidateRecord Center Label Slot RC)] [DecidableEq V] [MeasurableSpace Ω]

/-- Final consumer: raw fixed-center and B1/stability/availability premises actually
flow through studentization and into the reported interval probability. -/
theorem submission_complete_coverage {D : Type*} [Fintype D] [DecidableEq D]
    {I : ℕ → Center → Type*} [∀ n c, Fintype (I n c)] [∀ n c, Nonempty (I n c)]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (b : ReferenceData Ω Center D I) (hc : ReferenceConditions μ b)
    (a : LearningData Ω Context Center Label Slot RC EC V Parameter) (hl : LearningConditions μ a)
    (point_match : ∀ (n : ℕ) ω, (extend (a.fallbackC n ω) (a.reportC n ω)).point = b.point n ω)
    (variance_match : ∀ (n : ℕ) ω, (extend (a.fallbackC n ω) (a.reportC n ω)).variance = b.variance n ω)
    (σ : ℝ) (hσ : 0 < σ) (sigma_match : b.sigmaSq = σ^2)
    (rowCLT : TendstoInDistribution
      (fun (n : ℕ) ω => Real.sqrt ((n:ℝ)+1)*ErrorTarget.linearRowMean b.xi n ω/σ)
      atTop id (fun _ => μ) (gaussianReal 0 1))
    (hmM : ∀ n, AEMeasurable (b.rM n) μ)
    (hmSel : ∀ n, AEMeasurable (fun ω => a.pointA n ω-b.point n ω) μ)
    (hmV : ∀ n, AEMeasurable (a.varianceA n) μ)
    (z : ℝ) (hz : 0 ≤ z)
    (hmeas : ∀ n, MeasurableSet {ω | a.reportA n ω ≠ none ∧ 0 < a.varianceA n ω}) :
    Tendsto (fun n => μ.real {ω | (a.reportA n ω ≠ none ∧ 0 < a.varianceA n ω) ∧
      a.pointA n ω-z*Real.sqrt (a.varianceA n ω) ≤ b.target n ∧
      b.target n ≤ a.pointA n ω+z*Real.sqrt (a.varianceA n ω)}) atTop
      (𝓝 ((gaussianReal 0 1).real (Set.Icc (-z) z))) := by
  obtain ⟨hT, hV, hfail⟩ := submission_studentized μ b hc a hl point_match variance_match
    σ hσ sigma_match rowCLT hmM hmSel hmV
  exact submission_reported_coverage μ a.reportA a.pointA a.varianceA b.target
    σ z hσ hz hT hV hfail hmeas

/-- Parallel complete endpoint when whole centers, rather than individual rows,
are the independent asymptotic units.  Each `Fin (n+1)` contribution is one
center-aggregated full joint score.  The failure event remains in the coverage
denominator. -/
theorem submission_center_cluster_complete_coverage
    {D K : Type*} [Fintype D] [Fintype K]
    (P : Measure Ω) [IsProbabilityMeasure P]
    (report : Nat -> Ω -> Option Report)
    (actual actualVariance : Nat -> Ω -> Real) (target : Nat -> Real)
    (rM rSelection : Nat -> Ω -> Real)
    (a : Nat -> K -> Real)
    (H : Nat -> Matrix K D Real)
    (R : Nat -> Matrix D D Real)
    (g : (n : Nat) -> Ω -> Fin (n+1) -> D -> Real)
    (sigma z : Real) (hsigma : 0 < sigma) (hz : 0 <= z)
    (expansion : ∀ n w, actual n w - target n =
      ErrorTarget.linearRowMean
        (BinaryFormal.CenterClusterInference.centerInfluenceArray a H R g) n w +
        rM n w + rSelection n w)
    (centerCLT : TendstoInDistribution
      (fun (n : Nat) w => Real.sqrt ((n : Real)+1) *
        ErrorTarget.linearRowMean
          (BinaryFormal.CenterClusterInference.centerInfluenceArray a H R g) n w / sigma)
      atTop id (fun _ => P) (gaussianReal 0 1))
    (remainder_small : TendstoInMeasure P
      (fun (n : Nat) w => Real.sqrt ((n : Real)+1) * rM n w / sigma) atTop 0)
    (selection_small : TendstoInMeasure P
      (fun (n : Nat) w => Real.sqrt ((n : Real)+1) * rSelection n w / sigma) atTop 0)
    (measurable_remainder : ∀ n, AEMeasurable (rM n) P)
    (measurable_selection : ∀ n, AEMeasurable (rSelection n) P)
    (center_variance_consistent : TendstoInMeasure P
      (fun (n : Nat) w => (n+1) *
        BinaryFormal.CenterClusterInference.centerReferenceVarianceArray a H R g n w)
      atTop (fun _ => sigma^2))
    (actual_variance_close : TendstoInMeasure P
      (fun (n : Nat) w => (n+1) * (actualVariance n w -
        BinaryFormal.CenterClusterInference.centerReferenceVarianceArray a H R g n w))
      atTop 0)
    (measurable_variance : ∀ n, AEMeasurable (actualVariance n) P)
    (failure_zero : Tendsto (fun n => P {w | report n w = none}) atTop (𝓝 0))
    (measurable_available : ∀ n,
      MeasurableSet {w | report n w ≠ none ∧ 0 < actualVariance n w}) :
    Tendsto (fun n => P.real {w | (report n w ≠ none ∧ 0 < actualVariance n w) ∧
      actual n w-z*Real.sqrt (actualVariance n w) <= target n ∧
      target n <= actual n w+z*Real.sqrt (actualVariance n w)}) atTop
      (𝓝 ((gaussianReal 0 1).real (Set.Icc (-z) z))) := by
  have hT := BinaryFormal.CenterClusterInference.center_cluster_studentized
    P actual target rM rSelection actualVariance a H R g sigma hsigma
    expansion centerCLT remainder_small selection_small measurable_remainder
    measurable_selection center_variance_consistent actual_variance_close
    measurable_variance
  have hV := ErrorTarget.variance_consistency_transfer P
    (fun (n : Nat) w => (n+1) *
      BinaryFormal.CenterClusterInference.centerReferenceVarianceArray a H R g n w)
    (fun (n : Nat) w => (n+1) * actualVariance n w) (sigma^2)
    center_variance_consistent (by simpa only [mul_sub] using actual_variance_close)
  exact submission_reported_coverage P report actual actualVariance target
    sigma z hsigma hz hT hV failure_zero measurable_available
end Complete

#print axioms submission_complete_coverage
#print axioms submission_center_cluster_complete_coverage
#print axioms ReferenceData
#print axioms ReferenceData.u
#print axioms ReferenceData.v
#print axioms ReferenceData.s
#print axioms ReferenceData.r
#print axioms ReferenceData.delta
#print axioms ReferenceData.point
#print axioms ReferenceData.target
#print axioms ReferenceData.rM
#print axioms ReferenceData.scaledVariance
#print axioms ReferenceData.variance
#print axioms ReferenceData.sigmaSq
#print axioms ReferenceData.derivativeAt
#print axioms ReferenceConditions
#print axioms LearningData
#print axioms LearningData.reportA
#print axioms LearningData.reportC
#print axioms LearningData.pointA
#print axioms LearningData.varianceA
#print axioms LearningConditions
end BinarySubmission
