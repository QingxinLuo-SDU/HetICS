import BinaryCore
import FiniteFreeze
import BiostatFormal.HighDimensional.KnockoffAdjustment
import BiostatFormal.CausalInference.LabelCausalBridge
import Mathlib.MeasureTheory.Function.ConditionalExpectation.Basic
import Mathlib.MeasureTheory.Integral.Bochner.Basic

/-! Post-ICP screening, deterministic label transport and common-population identification.
The conditional mean assumptions below are supplied causal semantics, not properties proved
for the empirical DGP. Covariate integration has no finite-type restriction. -/

noncomputable section
open scoped BigOperators
open MeasureTheory Filter
open BiostatFormal.HighDimensional.Multiview
open BiostatFormal.CausalInference
open BiostatFormal.HighDimensional.KnockoffAdjustment

namespace BinaryFormal.ScreeningAdjustment

variable {Ω V : Type*} [DecidableEq V]

def registeredCore (Z shared specific : Finset V) : Finset V := Z ∪ shared ∪ specific

inductive Channel where
  | treatment | outcome | dual
  deriving DecidableEq

def candidates (channel : Channel) (CX CY : Finset V) : Finset V :=
  match channel with
  | .treatment => CX
  | .outcome => CY
  | .dual => CX ∪ CY

/-- The official ICP call occurs after the channel's candidate union. -/
def finalSelection (T : Finset V → Finset V) (channel : Channel)
    (CX CY : Finset V) : Finset V := T (candidates channel CX CY)

theorem finalSelection_subset (T : Finset V → Finset V)
    (hT : ∀ C, T C ⊆ C) (channel : Channel) (CX CY : Finset V) :
    finalSelection T channel CX CY ⊆ candidates channel CX CY :=
  hT _

def deletedEvent (C S : Ω → Finset V) (j : V) : Set Ω :=
  {ω | j ∈ C ω ∧ j ∉ S ω}

theorem final_miss_eq_candidate_miss_union_deleted
    (C S : Ω → Finset V) (hSC : ∀ ω, S ω ⊆ C ω) (j : V) :
    variableMissEvent S j = variableMissEvent C j ∪ deletedEvent C S j := by
  ext ω
  change (j ∉ S ω) ↔ (j ∉ C ω) ∨ (j ∈ C ω ∧ j ∉ S ω)
  constructor
  · intro hs
    by_cases hc : j ∈ C ω
    · exact Or.inr ⟨hc, hs⟩
    · exact Or.inl hc
  · rintro (hc | ⟨_, hs⟩)
    · exact fun hs => hc (hSC ω hs)
    · exact hs

theorem candidate_miss_disjoint_deleted (C S : Ω → Finset V) (j : V) :
    Disjoint (variableMissEvent C j) (deletedEvent C S j) := by
  apply Set.disjoint_left.mpr
  intro ω hc hd
  exact hc hd.1

theorem measure_final_miss_eq [MeasurableSpace Ω] (μ : Measure Ω)
    (C S : Ω → Finset V) (hSC : ∀ ω, S ω ⊆ C ω) (j : V)
    (hD : MeasurableSet (deletedEvent C S j)) :
    μ (variableMissEvent S j) =
      μ (variableMissEvent C j) + μ (deletedEvent C S j) := by
  rw [final_miss_eq_candidate_miss_union_deleted C S hSC j]
  exact measure_union (candidate_miss_disjoint_deleted C S j) hD

/-- Finite union bound under a single measure; no independence is needed. -/
theorem final_coverage_failure_le [MeasurableSpace Ω] (μ : Measure Ω)
    (core : Finset V) (C S : Ω → Finset V) (hSC : ∀ ω, S ω ⊆ C ω) :
    μ (sureScreeningEvent core S)ᶜ ≤
      (∑ j ∈ core, μ (variableMissEvent C j)) +
        ∑ j ∈ core, μ (deletedEvent C S j) := by
  calc
    _ ≤ ∑ j ∈ core, μ (variableMissEvent S j) :=
      measure_sureScreeningFailure_le_sum_miss μ core S
    _ ≤ ∑ j ∈ core, (μ (variableMissEvent C j) + μ (deletedEvent C S j)) := by
      apply Finset.sum_le_sum
      intro j hj
      rw [final_miss_eq_candidate_miss_union_deleted C S hSC j]
      exact measure_union_le _ _
    _ = _ := Finset.sum_add_distrib

omit [DecidableEq V] in
theorem deletedEvent_identity (C : Ω → Finset V) (j : V) :
    deletedEvent C C j = ∅ := by
  ext ω
  simp [deletedEvent]

theorem single_center_coverage_failure_le [MeasurableSpace Ω] (μ : Measure Ω)
    (core : Finset V) (C : Ω → Finset V) :
    μ (sureScreeningEvent core C)ᶜ ≤ ∑ j ∈ core, μ (variableMissEvent C j) := by
  have h := final_coverage_failure_le μ core C C (fun _ => Finset.Subset.refl _)
  simpa [deletedEvent_identity] using h

theorem multiclass_coverage_failure_le {K : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (classes : Finset K) (core : K → Finset V)
    (C S : K → Ω → Finset V) (hSC : ∀ k ω, S k ω ⊆ C k ω) :
    μ (⋃ k ∈ classes, (sureScreeningEvent (core k) (S k))ᶜ) ≤
      ∑ k ∈ classes, ((∑ j ∈ core k, μ (variableMissEvent (C k) j)) +
        ∑ j ∈ core k, μ (deletedEvent (C k) (S k) j)) := by
  exact (measure_biUnion_finset_le classes _).trans
    (Finset.sum_le_sum fun k _ => final_coverage_failure_le μ (core k) (C k) (S k) (hSC k))

theorem final_valid_of_coverage_safe (core S : Finset V) (safe : Finset V → Prop)
    (hCoverage : core ⊆ S) (hSafe : safe S) : ValidAdjustment core safe S :=
  validAdjustment_of_complete_of_safe hCoverage hSafe

theorem final_baseline_only (baseline C S : Finset V)
    (hC : CandidatePoolBaseline baseline C) (hS : S ⊆ C) : BaselineOnly baseline S :=
  baselineOnly_of_selected_subset_candidates hC hS

/-- Deterministic FDP, with the standard denominator convention at empty selection. -/
def falseDiscoveryProportion (nonnull S : Finset V) : ℝ :=
  ((S \ nonnull).card : ℝ) / max (S.card : ℝ) 1

theorem empty_fdp_zero_misses_core (j : V) :
    falseDiscoveryProportion ({j} : Finset V) ∅ = 0 ∧
      ¬ CompleteAdjustment ({j} : Finset V) ∅ := by
  simp [falseDiscoveryProportion, CompleteAdjustment]

section LabelTransport
variable {Center Label Context Row : Type*}

/-- Data, all seeds, order and numerical rules belong to the context/closed-over functions.
The core is not an explicit rule argument. As Context is arbitrary, that it contains only
observed inputs/seeds/rules is a program-interface premise, not a no-truth-leakage theorem. -/
def finalRule (candidate : Context → Set Center → Finset V)
    (icp : Context → Set Center → Finset V → Finset V)
    (context : Context) (members : Set Center) : Finset V :=
  icp context members (candidate context members)

def finalAdjustment (candidate : Context → Set Center → Finset V)
    (icp : Context → Set Center → Finset V → Finset V)
    (context : Context) (labels : Partition Center Label) (label : Label) : Finset V :=
  classwiseAdjustment (finalRule candidate icp context) labels label

/-- Public B1 consumer: pass its Exact output and the same complete candidate+ICP rule. -/
theorem final_sets_eq_under_exact
    (candidate : Context → Set Center → Finset V)
    (icp : Context → Set Center → Finset V → Finset V)
    (context : Context) {estimated truth : Partition Center Label}
    (hExact : Exact estimated truth) :
    ∃ σ : LabelPermutation Label, ∀ label,
      finalAdjustment candidate icp context estimated label =
        finalAdjustment candidate icp context truth (σ label) :=
  classwiseAdjustment_eq_under_exact (finalRule candidate icp context) hExact

theorem final_sets_eq_under_exact_equal_contexts
    (candidate : Context → Set Center → Finset V)
    (icp : Context → Set Center → Finset V → Finset V)
    (estimatedContext truthContext : Context) (hContext : estimatedContext = truthContext)
    {estimated truth : Partition Center Label} (hExact : Exact estimated truth) :
    ∃ σ : LabelPermutation Label, ∀ label,
      finalAdjustment candidate icp estimatedContext estimated label =
        finalAdjustment candidate icp truthContext truth (σ label) := by
  subst truthContext
  exact final_sets_eq_under_exact candidate icp estimatedContext hExact

theorem validAdjustment_transport {core S S' : Finset V} {safe : Finset V → Prop}
    (hSame : S = S') (hValid : ValidAdjustment core safe S') :
    ValidAdjustment core safe S := by
  simpa only [hSame] using hValid

def classRows [DecidableEq Row] [DecidableEq Label] (rows : Finset Row)
    (center : Row → Center) (labels : Partition Center Label) (label : Label) : Finset Row :=
  rows.filter fun i => labels (center i) = label

theorem classRows_eq_of_members_eq [DecidableEq Row] [DecidableEq Label]
    (rows : Finset Row) (center : Row → Center)
    {estimated truth : Partition Center Label} {l l' : Label}
    (hMembers : classMembers estimated l = classMembers truth l') :
    classRows rows center estimated l = classRows rows center truth l' := by
  ext i
  have hi := Set.ext_iff.mp hMembers (center i)
  simpa only [classRows, Finset.mem_filter, classMembers, Set.mem_setOf_eq] using
    and_congr_right (fun _ => hi)

/-- Targets indexed by member sets must use the same underlying reference population law. -/
def memberWeightedTarget [DecidableEq Row] [DecidableEq Label] [Fintype Label]
    (rows : Finset Row) (center : Row → Center) (labels : Partition Center Label)
    (target : Set Center → ℝ) : ℝ :=
  ∑ l, ((classRows rows center labels l).card : ℝ) / rows.card * target (classMembers labels l)

theorem exact_final_sets_counts_and_target
    [DecidableEq Row] [DecidableEq Label] [Fintype Label]
    (rows : Finset Row) (center : Row → Center)
    (candidate : Context → Set Center → Finset V)
    (icp : Context → Set Center → Finset V → Finset V)
    (context : Context) (target : Set Center → ℝ)
    {estimated truth : Partition Center Label} (hExact : Exact estimated truth) :
    (∃ σ : LabelPermutation Label, ∀ l,
      classRows rows center estimated l = classRows rows center truth (σ l) ∧
      finalAdjustment candidate icp context estimated l =
        finalAdjustment candidate icp context truth (σ l)) ∧
    memberWeightedTarget rows center estimated target =
      memberWeightedTarget rows center truth target := by
  obtain ⟨σ, hMembers⟩ := classMembers_eq_under_exact hExact
  have hRows (l : Label) := classRows_eq_of_members_eq rows center (hMembers l)
  constructor
  · refine ⟨σ, fun l => ⟨hRows l, ?_⟩⟩
    change finalRule candidate icp context (classMembers estimated l) =
      finalRule candidate icp context (classMembers truth (σ l))
    rw [hMembers l]
  · unfold memberWeightedTarget
    calc
      _ = ∑ l, ((classRows rows center truth (σ l)).card : ℝ) / rows.card *
          target (classMembers truth (σ l)) := by
        apply Finset.sum_congr rfl
        intro l _
        rw [hRows l, hMembers l]
      _ = _ := Equiv.sum_comp σ (fun l =>
        ((classRows rows center truth l).card : ℝ) / rows.card * target (classMembers truth l))
end LabelTransport

section GeneralIntegral
variable {X : Type*} [MeasurableSpace X]

def gIntegrand (b : X → BinaryCore.BinaryLaw) (x : X) : ℝ := (b x).mu1 - (b x).mu0
def aipwIntegrand (b : X → BinaryCore.BinaryLaw) (e m0 m1 : X → ℝ) (x : X) : ℝ :=
  (b x).fourStateMean (e x) (m0 x) (m1 x)
def biasIntegrand (b : X → BinaryCore.BinaryLaw) (e m0 m1 : X → ℝ) (x : X) : ℝ :=
  ((b x).e0 - e x) * (((b x).mu1 - m1 x) / e x + ((b x).mu0 - m0 x) / (1 - e x))
/-- Integral of the four-state conditional mean. Equality to the full observed score's
expectation additionally needs its conditional-law version and L1 integrability; integrability
of this conditional mean alone does not imply integrability of the full observed score. -/
def populationAIPW (ν : Measure X) (b : X → BinaryCore.BinaryLaw) (e m0 m1 : X → ℝ) : ℝ :=
  ∫ x, aipwIntegrand b e m0 m1 x ∂ν
def gFormula (ν : Measure X) (b : X → BinaryCore.BinaryLaw) : ℝ :=
  ∫ x, gIntegrand b x ∂ν

theorem aipw_ae_eq_g_add_bias (ν : Measure X) (b : X → BinaryCore.BinaryLaw)
    (e m0 m1 : X → ℝ) (hDomain : ∀ᵐ x ∂ν, 0 < e x ∧ e x < 1) :
    aipwIntegrand b e m0 m1 =ᵐ[ν] fun x => gIntegrand b x + biasIntegrand b e m0 m1 x := by
  filter_upwards [hDomain] with x hx
  exact (b x).four_state_bias (e x) (m0 x) (m1 x) hx.1 hx.2

theorem aipw_integrable (ν : Measure X) (b : X → BinaryCore.BinaryLaw)
    (e m0 m1 : X → ℝ) (hDomain : ∀ᵐ x ∂ν, 0 < e x ∧ e x < 1)
    (hG : Integrable (gIntegrand b) ν) (hB : Integrable (biasIntegrand b e m0 m1) ν) :
    Integrable (aipwIntegrand b e m0 m1) ν :=
  (hG.add hB).congr (aipw_ae_eq_g_add_bias ν b e m0 m1 hDomain).symm

/-- True Bochner integration over an arbitrary covariate measure, not a finite sum. -/
theorem population_aipw_bias (ν : Measure X) (b : X → BinaryCore.BinaryLaw)
    (e m0 m1 : X → ℝ) (hDomain : ∀ᵐ x ∂ν, 0 < e x ∧ e x < 1)
    (hG : Integrable (gIntegrand b) ν) (hB : Integrable (biasIntegrand b e m0 m1) ν) :
    populationAIPW ν b e m0 m1 - gFormula ν b = ∫ x, biasIntegrand b e m0 m1 x ∂ν := by
  have hEq := integral_congr_ae (aipw_ae_eq_g_add_bias ν b e m0 m1 hDomain)
  rw [integral_add hG hB] at hEq
  exact sub_eq_iff_eq_add.mpr (by simpa [populationAIPW, gFormula, add_comm] using hEq)

theorem population_aipw_dr (ν : Measure X) (b : X → BinaryCore.BinaryLaw)
    (e m0 m1 : X → ℝ) (hDomain : ∀ᵐ x ∂ν, 0 < e x ∧ e x < 1)
    (hG : Integrable (gIntegrand b) ν)
    (hCorrect : (∀ᵐ x ∂ν, e x = (b x).e0) ∨
      (∀ᵐ x ∂ν, m0 x = (b x).mu0 ∧ m1 x = (b x).mu1)) :
    Integrable (aipwIntegrand b e m0 m1) ν ∧ populationAIPW ν b e m0 m1 = gFormula ν b := by
  have hz : biasIntegrand b e m0 m1 =ᵐ[ν] (fun _ => (0 : ℝ)) := by
    rcases hCorrect with hp | ho
    · filter_upwards [hp] with x hx
      simp [biasIntegrand, hx]
    · filter_upwards [ho] with x hx
      simp [biasIntegrand, hx.1, hx.2]
  have hB : Integrable (biasIntegrand b e m0 m1) ν := (integrable_zero X ℝ ν).congr hz.symm
  refine ⟨aipw_integrable ν b e m0 m1 hDomain hG hB, ?_⟩
  have h := population_aipw_bias ν b e m0 m1 hDomain hG hB
  rw [integral_congr_ae hz, integral_zero] at h
  exact sub_eq_zero.mp h
end GeneralIntegral

section CommonPotentialOutcomes
variable [mΩ : MeasurableSpace Ω]

def commonATE (μ : Measure Ω) (Y0 Y1 : Ω → ℝ) : ℝ := ∫ ω, Y1 ω - Y0 ω ∂μ

/-- Mean-level causal source interface. q a t is the potential mean for target arm a
given realized arm t and the selected covariates. Its causal interpretation is supplied;
the equalities are explicit consistency, exchangeability and conditional-mean versions.
The same Y0/Y1 must be used for every qualified selected set. -/
structure MeanIdentification (μ : Measure Ω) (Y0 Y1 : Ω → ℝ)
    (b : Ω → BinaryCore.BinaryLaw) where
  covariateSigma : MeasurableSpace Ω
  sigma_le : covariateSigma ≤ mΩ
  sigmaFinite : SigmaFinite (μ.trim sigma_le)
  potentialMean : Bool → Ω → ℝ
  potentialGivenTreatment : Bool → Bool → Ω → ℝ
  consistency0 : (fun ω => (b ω).mu0) =ᵐ[μ] potentialGivenTreatment false false
  consistency1 : (fun ω => (b ω).mu1) =ᵐ[μ] potentialGivenTreatment true true
  exchangeability : ∀ a t, potentialGivenTreatment a t =ᵐ[μ] potentialMean a
  version0 : potentialMean false =ᵐ[μ] μ[Y0 | covariateSigma]
  version1 : potentialMean true =ᵐ[μ] μ[Y1 | covariateSigma]
  truePositivity : ∀ᵐ ω ∂μ, 0 < (b ω).e0 ∧ (b ω).e0 < 1

theorem observed_means_eq_condExp (μ : Measure Ω) (Y0 Y1 : Ω → ℝ)
    (b : Ω → BinaryCore.BinaryLaw) (h : MeanIdentification μ Y0 Y1 b) :
    ((fun ω => (b ω).mu0) =ᵐ[μ] μ[Y0 | h.covariateSigma]) ∧
    ((fun ω => (b ω).mu1) =ᵐ[μ] μ[Y1 | h.covariateSigma]) :=
  ⟨h.consistency0.trans ((h.exchangeability false false).trans h.version0),
   h.consistency1.trans ((h.exchangeability true true).trans h.version1)⟩

/-- Uses mathlib's actual conditional expectation integral theorem (tower at the mean level). -/
theorem gformula_identifies_common_ate (μ : Measure Ω) (Y0 Y1 : Ω → ℝ)
    (hY0 : Integrable Y0 μ) (hY1 : Integrable Y1 μ)
    (b : Ω → BinaryCore.BinaryLaw) (h : MeanIdentification μ Y0 Y1 b) :
    Integrable (gIntegrand b) μ ∧ gFormula μ b = commonATE μ Y0 Y1 := by
  letI := h.sigmaFinite
  obtain ⟨h0, h1⟩ := observed_means_eq_condExp μ Y0 Y1 b h
  have hi0 : Integrable (fun ω => (b ω).mu0) μ := integrable_condExp.congr h0.symm
  have hi1 : Integrable (fun ω => (b ω).mu1) μ := integrable_condExp.congr h1.symm
  refine ⟨hi1.sub hi0, ?_⟩
  unfold gFormula gIntegrand commonATE
  rw [integral_sub hi1 hi0, integral_sub hY1 hY0,
    integral_congr_ae h1, integral_congr_ae h0,
    integral_condExp h.sigma_le, integral_condExp h.sigma_le]

theorem aipw_identifies_common_ate (μ : Measure Ω) (Y0 Y1 : Ω → ℝ)
    (hY0 : Integrable Y0 μ) (hY1 : Integrable Y1 μ)
    (b : Ω → BinaryCore.BinaryLaw) (h : MeanIdentification μ Y0 Y1 b)
    (e m0 m1 : Ω → ℝ) (hDomain : ∀ᵐ ω ∂μ, 0 < e ω ∧ e ω < 1)
    (hCorrect : (∀ᵐ ω ∂μ, e ω = (b ω).e0) ∨
      (∀ᵐ ω ∂μ, m0 ω = (b ω).mu0 ∧ m1 ω = (b ω).mu1)) :
    populationAIPW μ b e m0 m1 = commonATE μ Y0 Y1 := by
  obtain ⟨hG, hTarget⟩ := gformula_identifies_common_ate μ Y0 Y1 hY0 hY1 b h
  exact (population_aipw_dr μ b e m0 m1 hDomain hG hCorrect).2.trans hTarget

/-- Completeness+safety grants mean-identification premises, not the target equality.
The target random variables and measure are outside the selected-set index. -/
structure CommonAdjustmentBridge (μ : Measure Ω) (Y0 Y1 : Ω → ℝ)
    (core : Finset V) (safe : Finset V → Prop) where
  lawFor : Finset V → Ω → BinaryCore.BinaryLaw
  conditions_of_valid : ∀ S, ValidAdjustment core safe S →
    MeanIdentification μ Y0 Y1 (lawFor S)

theorem valid_gformula_common_target (μ : Measure Ω) (Y0 Y1 : Ω → ℝ)
    (hY0 : Integrable Y0 μ) (hY1 : Integrable Y1 μ)
    (core : Finset V) (safe : Finset V → Prop)
    (bridge : CommonAdjustmentBridge μ Y0 Y1 core safe)
    (S : Finset V) (hValid : ValidAdjustment core safe S) :
    gFormula μ (bridge.lawFor S) = commonATE μ Y0 Y1 :=
  (gformula_identifies_common_ate μ Y0 Y1 hY0 hY1 _
    (bridge.conditions_of_valid S hValid)).2

theorem different_valid_sets_same_target (μ : Measure Ω) (Y0 Y1 : Ω → ℝ)
    (hY0 : Integrable Y0 μ) (hY1 : Integrable Y1 μ)
    (core : Finset V) (safe : Finset V → Prop)
    (bridge : CommonAdjustmentBridge μ Y0 Y1 core safe)
    (S T : Finset V) (hS : ValidAdjustment core safe S) (hT : ValidAdjustment core safe T) :
    gFormula μ (bridge.lawFor S) = gFormula μ (bridge.lawFor T) :=
  (valid_gformula_common_target μ Y0 Y1 hY0 hY1 core safe bridge S hS).trans
    (valid_gformula_common_target μ Y0 Y1 hY0 hY1 core safe bridge T hT).symm
end CommonPotentialOutcomes

section PersonWeights
variable {I K : Type*} [DecidableEq I] [DecidableEq K] [Fintype K]

/-- p0/p1 are numeric expectations under ONE reference population law; they are
not automatically the generator's realized, covariate-conditional probabilities. -/
theorem person_weighted_reference_means (p : BinaryCore.SamplePartition I K)
    (p0 p1 : I → ℝ) :
    (∑ k, p.weight k * p.clusterMean (fun i => p1 i - p0 i) k) =
      p.fullMean (fun i => p1 i - p0 i) :=
  p.weighted_mean_eq_full _

/-- Each class's identified value must equal its mean of the common reference values.
The equalities may be supplied by the common-potential-outcome identification theorem. -/
theorem weighted_identified_targets (p : BinaryCore.SamplePartition I K)
    (p0 p1 : I → ℝ) (identified : K → ℝ)
    (h : ∀ k, identified k = p.clusterMean (fun i => p1 i - p0 i) k) :
    (∑ k, p.weight k * identified k) = p.fullMean (fun i => p1 i - p0 i) := by
  simp_rw [h]
  exact person_weighted_reference_means p p0 p1

/-- Fixed reference class population laws; heterogeneous class ATEs remain distinct.
Person counts specify the target population composition, exactly as in Task 2. -/
theorem person_weighted_aipw_common_population [MeasurableSpace Ω]
    (p : BinaryCore.SamplePartition I K) (μ : K → Measure Ω)
    (Y0 Y1 : K → Ω → ℝ)
    (hY0 : ∀ k, Integrable (Y0 k) (μ k)) (hY1 : ∀ k, Integrable (Y1 k) (μ k))
    (b : K → Ω → BinaryCore.BinaryLaw)
    (hIdentification : ∀ k, MeanIdentification (μ k) (Y0 k) (Y1 k) (b k))
    (e m0 m1 : K → Ω → ℝ)
    (hDomain : ∀ k, ∀ᵐ ω ∂μ k, 0 < e k ω ∧ e k ω < 1)
    (hCorrect : ∀ k, (∀ᵐ ω ∂μ k, e k ω = (b k ω).e0) ∨
      (∀ᵐ ω ∂μ k, m0 k ω = (b k ω).mu0 ∧ m1 k ω = (b k ω).mu1)) :
    (∑ k, p.weight k * populationAIPW (μ k) (b k) (e k) (m0 k) (m1 k)) =
      ∑ k, p.weight k * commonATE (μ k) (Y0 k) (Y1 k) := by
  apply Finset.sum_congr rfl
  intro k _
  rw [aipw_identifies_common_ate (μ k) (Y0 k) (Y1 k) (hY0 k) (hY1 k)
    (b k) (hIdentification k) (e k) (m0 k) (m1 k) (hDomain k) (hCorrect k)]
end PersonWeights

#print axioms registeredCore
#print axioms Channel
#print axioms candidates
#print axioms finalSelection
#print axioms finalSelection_subset
#print axioms deletedEvent
#print axioms final_miss_eq_candidate_miss_union_deleted
#print axioms candidate_miss_disjoint_deleted
#print axioms measure_final_miss_eq
#print axioms final_coverage_failure_le
#print axioms deletedEvent_identity
#print axioms single_center_coverage_failure_le
#print axioms multiclass_coverage_failure_le
#print axioms final_valid_of_coverage_safe
#print axioms final_baseline_only
#print axioms falseDiscoveryProportion
#print axioms empty_fdp_zero_misses_core
#print axioms finalRule
#print axioms finalAdjustment
#print axioms final_sets_eq_under_exact
#print axioms final_sets_eq_under_exact_equal_contexts
#print axioms validAdjustment_transport
#print axioms classRows
#print axioms classRows_eq_of_members_eq
#print axioms memberWeightedTarget
#print axioms exact_final_sets_counts_and_target
#print axioms gIntegrand
#print axioms aipwIntegrand
#print axioms biasIntegrand
#print axioms populationAIPW
#print axioms gFormula
#print axioms aipw_ae_eq_g_add_bias
#print axioms aipw_integrable
#print axioms population_aipw_bias
#print axioms population_aipw_dr
#print axioms commonATE
#print axioms MeanIdentification
#print axioms observed_means_eq_condExp
#print axioms gformula_identifies_common_ate
#print axioms aipw_identifies_common_ate
#print axioms CommonAdjustmentBridge
#print axioms valid_gformula_common_target
#print axioms different_valid_sets_same_target
#print axioms person_weighted_reference_means
#print axioms weighted_identified_targets
#print axioms person_weighted_aipw_common_population

end BinaryFormal.ScreeningAdjustment
