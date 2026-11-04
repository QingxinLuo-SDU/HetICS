import FiniteFreezeConsumer
import BiostatFormal.HighDimensional.Multiview.B2V3A3Connections

/-! Current strict selection composed with audited source survival and frozen output.
Source-field correspondence and statistical tails remain explicit inputs. -/
namespace BinaryFormal.B1Current
open BiostatFormal.HighDimensional.Multiview
open BinaryFormal.FiniteFreeze MeasureTheory
noncomputable section

variable {C : Type*}

abbrev LogicalOrder (C : Type*) :=
  List (Option (RawDirection C) × Option (RawDirection C))

/-- Raw field correspondence: both directional records and permitted families exist;
one direction retains this complete payload and has a finite accepted risk value. -/
def SubmittedFields (logical : LogicalOrder C) (c : C) : Prop :=
  ∃ l r, (some l, some r) ∈ logical ∧ l.familyAllowed = true ∧
    r.familyAllowed = true ∧
    ((l.candidate = c ∧ ∃ v, l.risk = some v) ∨
     (r.candidate = c ∧ ∃ v, r.risk = some v))

theorem submittedFields_mem [DecidableEq C] (logical : LogicalOrder C) (c : C)
    (h : SubmittedFields logical c) : c ∈ orderedPool (primaryOrder logical) := by
  rcases h with ⟨l, r, hpair, hl, hr, hc⟩
  have hin : ∃ v, (c, v) ∈
      (logical.flatMap fun p => logicalPair p.1 p.2).filterMap id := by
    rcases hc with ⟨he, v, hv⟩ | ⟨he, v, hv⟩
    · refine ⟨v, List.mem_filterMap.mpr ?_⟩
      refine ⟨some (c, v), List.mem_flatMap.mpr ?_, rfl⟩
      exact ⟨(some l, some r), hpair, by simp [logicalPair, hl, hr, scoredDirection, hv, he]⟩
    · refine ⟨v, List.mem_filterMap.mpr ?_⟩
      refine ⟨some (c, v), List.mem_flatMap.mpr ?_, rfl⟩
      exact ⟨(some l, some r), hpair, by simp [logicalPair, hl, hr, scoredDirection, hv, he]⟩
  obtain ⟨v, hv⟩ := hin
  simp only [orderedPool, primaryOrder, List.mem_toFinset, List.mem_filterMap,
    List.mem_map, id_eq, exists_exists_and_eq_and]
  exact ⟨(c, v), List.mem_filterMap.mp hv, rfl⟩

theorem sourcePool_subset_current [DecidableEq C]
    (pool : GuardedCandidatePool C) (logical : LogicalOrder C)
    (hfields : ∀ c ∈ pool, SubmittedFields logical c) :
    pool ⊆ orderedPool (primaryOrder logical) := by
  intro c hc
  exact submittedFields_mem logical c (hfields c hc)

/-- Shared population/empirical weights: Y/2 + X/2; F and stability have weight zero. -/
def primaryRiskPair [RiskInput C] (popY popX empY empX : C → ℝ) : CompositeRiskPair C where
  weight := fun k => match k with
    | .outcome | .treatment => 1 / 2
    | .covariate | .stability => 0
  populationComponent := fun k => match k with
    | .outcome => popY | .treatment => popX | _ => fun _ => 0
  empiricalComponent := fun k => match k with
    | .outcome => empY | .treatment => empX | _ => fun _ => 0

theorem primaryRiskPair_values [RiskInput C] (popY popX empY empX : C → ℝ) (c : C) :
    populationCompositeRisk (primaryRiskPair popY popX empY empX) c =
      (popY c + popX c) / 2 ∧
    empiricalCompositeRisk (primaryRiskPair popY popX empY empX) c =
      (empY c + empX c) / 2 := by
  have hu : (Finset.univ : Finset RiskComponent) =
      {.outcome, .treatment, .covariate, .stability} := by decide
  simp [populationCompositeRisk, empiricalCompositeRisk, hu, primaryRiskPair]
  constructor <;> ring

/-- Only the four weights are constrained; diagnostic component functions remain arbitrary. -/
theorem primaryRisk_values [RiskInput C] (risk : CompositeRiskPair C)
    (hY : risk.weight .outcome = 1 / 2) (hX : risk.weight .treatment = 1 / 2)
    (hF : risk.weight .covariate = 0) (hS : risk.weight .stability = 0) (c : C) :
    populationCompositeRisk risk c =
      (risk.populationComponent .outcome c + risk.populationComponent .treatment c) / 2 ∧
    empiricalCompositeRisk risk c =
      (risk.empiricalComponent .outcome c + risk.empiricalComponent .treatment c) / 2 := by
  have hu : (Finset.univ : Finset RiskComponent) =
      {.outcome, .treatment, .covariate, .stability} := by decide
  simp [populationCompositeRisk, empiricalCompositeRisk, hu, hY, hX, hF, hS]
  constructor <;> ring

section Generic
variable {Omega Center Label Slot RC EC : Type*}
  [Fintype Center] [DecidableEq Center] [Fintype Label] [DecidableEq Label]
  [DecidableEq (CandidateRecord Center Label Slot RC)]

abbrev currentPool (ordered : Omega → List (Option (CandidateRecord Center Label Slot RC))) :=
  fun ω => orderedPool (ordered ω)

abbrev currentOutput (ordered : Omega → List (Option (CandidateRecord Center Label Slot RC)))
    (spec : SelectorRiskSpec (CandidateRecord Center Label Slot RC) EC) :=
  fun ω => orderedOutput (empiricalCompositeRisk spec.risk) (ordered ω)

theorem ordered_recovery (ordered : Omega → List (Option (CandidateRecord Center Label Slot RC)))
    (spec : SelectorRiskSpec (CandidateRecord Center Label Slot RC) EC)
    (truth : Partition Center Label) (u : ℝ) :
    exactSurvivalEvent (currentPool ordered) truth ∩
      uniformDeviationEvent (currentPool ordered) spec u ∩
      selectorStrictPopulationGapEvent (currentPool ordered) spec truth u 0 0 0 ⊆
      exactSelectedEvent (currentOutput ordered spec) truth := by
  intro ω h
  apply recoverySemanticCore (currentPool ordered) spec (currentOutput ordered spec)
    truth u 0 0 0
  refine ⟨h, ?_⟩
  rw [ordered_selector_success_univ ordered spec truth]
  trivial

omit [Fintype Center] [DecidableEq Center] [Fintype Label] [DecidableEq Label] in
theorem survival_mono
    (source target : Omega → GuardedCandidatePool (CandidateRecord Center Label Slot RC))
    (truth : Partition Center Label) (h : ∀ ω, source ω ⊆ target ω) :
    exactSurvivalEvent source truth ⊆ exactSurvivalEvent target truth := by
  rintro ω ⟨c, hc⟩
  exact ⟨⟨c.1, h ω c.2⟩, hc⟩

theorem failure_subset
    (ordered : Omega → List (Option (CandidateRecord Center Label Slot RC)))
    (spec : SelectorRiskSpec (CandidateRecord Center Label Slot RC) EC)
    (truth : Partition Center Label) (u : ℝ)
    (source : Omega → GuardedCandidatePool (CandidateRecord Center Label Slot RC))
    (coverage : Set Omega) (hpool : ∀ ω, source ω ⊆ currentPool ordered ω) :
    (exactSelectedEvent (currentOutput ordered spec) truth)ᶜ ⊆
      coverageᶜ ∪ (coverage ∩ (exactSurvivalEvent source truth)ᶜ) ∪
        (uniformDeviationEvent (currentPool ordered) spec u)ᶜ ∪
        (selectorStrictPopulationGapEvent (currentPool ordered) spec truth u 0 0 0)ᶜ := by
  classical
  intro ω hf
  by_cases hc : ω ∈ coverage
  · by_cases hs : ω ∈ exactSurvivalEvent source truth
    · by_cases hd : ω ∈ uniformDeviationEvent (currentPool ordered) spec u
      · by_cases hg : ω ∈ selectorStrictPopulationGapEvent (currentPool ordered) spec truth u 0 0 0
        · exact (hf (ordered_recovery ordered spec truth u
            ⟨⟨survival_mono source (currentPool ordered) truth hpool hs, hd⟩, hg⟩)).elim
        · exact Or.inr hg
      · exact Or.inl (Or.inr hd)
    · exact Or.inl (Or.inl (Or.inr ⟨hc, hs⟩))
  · exact Or.inl (Or.inl (Or.inl hc))

theorem failure_bound [MeasurableSpace Omega] (μ : Measure Omega)
    (ordered : Omega → List (Option (CandidateRecord Center Label Slot RC)))
    (spec : SelectorRiskSpec (CandidateRecord Center Label Slot RC) EC)
    (truth : Partition Center Label) (u : ℝ)
    (source : Omega → GuardedCandidatePool (CandidateRecord Center Label Slot RC))
    (coverage : Set Omega) (hpool : ∀ ω, source ω ⊆ currentPool ordered ω)
    (δa δr δd δg : ENNReal)
    (ha : μ coverageᶜ ≤ δa)
    (hr : μ (coverage ∩ (exactSurvivalEvent source truth)ᶜ) ≤ δr)
    (hd : μ (uniformDeviationEvent (currentPool ordered) spec u)ᶜ ≤ δd)
    (hg : μ (selectorStrictPopulationGapEvent (currentPool ordered) spec truth u 0 0 0)ᶜ ≤ δg) :
    μ (exactSelectedEvent (currentOutput ordered spec) truth)ᶜ ≤ δa + δr + δd + δg := by
  calc
    _ ≤ μ (coverageᶜ ∪ (coverage ∩ (exactSurvivalEvent source truth)ᶜ) ∪
        (uniformDeviationEvent (currentPool ordered) spec u)ᶜ ∪
        (selectorStrictPopulationGapEvent (currentPool ordered) spec truth u 0 0 0)ᶜ) :=
      measure_mono (failure_subset ordered spec truth u source coverage hpool)
    _ ≤ (μ coverageᶜ + μ (coverage ∩ (exactSurvivalEvent source truth)ᶜ)) +
        μ (uniformDeviationEvent (currentPool ordered) spec u)ᶜ +
        μ (selectorStrictPopulationGapEvent (currentPool ordered) spec truth u 0 0 0)ᶜ := by
      exact (measure_union_le _ _).trans (add_le_add
        ((measure_union_le _ _).trans (add_le_add (measure_union_le _ _) le_rfl)) le_rfl)
    _ ≤ _ := add_le_add (add_le_add (add_le_add ha hr) hd) hg

def frozenOutput {Parameter : Type*}
    (ordered : List (Option (CandidateRecord Center Label Slot RC)))
    (spec : SelectorRiskSpec (CandidateRecord Center Label Slot RC) EC)
    (refit : Option (Refit Center Label Parameter)) : Option (FrozenResult Center Label Parameter) :=
  freeze ((chooseOptional (empiricalCompositeRisk spec.risk) ordered).map fun c => c.label) refit

omit [Fintype Center] [DecidableEq Center] [Fintype Label] [DecidableEq Label] in
theorem exact_same_cluster (frozen truth : Partition Center Label) (h : Exact frozen truth)
    (i j : Center) : frozen i = frozen j ↔ truth i = truth j := by
  obtain ⟨σ, hσ⟩ := h
  rw [← hσ]
  exact σ.injective.eq_iff.symm

omit [Fintype Center] [DecidableEq Center] [Fintype Label] [DecidableEq Label] in
theorem frozen_selected_exact {Parameter : Type*}
    (ordered : Omega → List (Option (CandidateRecord Center Label Slot RC)))
    (spec : SelectorRiskSpec (CandidateRecord Center Label Slot RC) EC)
    (truth : Partition Center Label) (refit : Option (Refit Center Label Parameter))
    {ω : Omega} (h : ω ∈ exactSelectedEvent (currentOutput ordered spec) truth) :
    ∃ frozen, frozenOutput (ordered ω) spec refit =
      some ⟨frozen, usableRefit frozen refit⟩ ∧ Exact frozen truth ∧
      ∀ i j, frozen i = frozen j ↔ truth i = truth j := by
  obtain ⟨c, hc, he⟩ := h
  have hs := (orderedOutput_selected _ _ _).mp hc
  refine ⟨c.1.label, ?_, he, exact_same_cluster _ _ he⟩
  simp [frozenOutput, hs, freeze]

omit [Fintype Center] [DecidableEq Center] [Fintype Label] [DecidableEq Label]
  [DecidableEq (CandidateRecord Center Label Slot RC)] in
theorem frozen_abstain {Parameter : Type*}
    (ordered : List (Option (CandidateRecord Center Label Slot RC)))
    (spec : SelectorRiskSpec (CandidateRecord Center Label Slot RC) EC)
    (refit : Option (Refit Center Label Parameter))
    (h : chooseOptional (empiricalCompositeRisk spec.risk) ordered = none) :
    frozenOutput ordered spec refit = none := by simp [frozenOutput, h, freeze]

omit [Fintype Center] [DecidableEq Center] [Fintype Label] [DecidableEq Label] in
theorem usable_exact_rows {Parameter Row : Type*}
    (center : Row → Center) (frozen truth : Partition Center Label)
    (refit : Option (Refit Center Label Parameter))
    (hexact : Exact frozen truth) (p : Parameter) (status : Completion)
    (huse : usableRefit frozen refit = some (p, status)) :
    (∀ i j, rowLabel center frozen i = rowLabel center frozen j ↔
      truth (center i) = truth (center j)) ∧
    ∃ r q, refit = some r ∧ r.eligible = true ∧
      r.postLabel = frozen ∧ r.parameters = some q := by
  exact ⟨fun i j => exact_same_cluster frozen truth hexact (center i) (center j),
    (usableRefit_iff frozen refit).mp ⟨p, status, huse⟩⟩
end Generic

section Source
variable {E T : Nat} {Omega Fit Norm Frozen EC Key Owner RejectReason : Type*}
  [MeasurableSpace Omega] [SeminormedAddCommGroup Fit] [Nonempty (Fin E)]
  [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
  (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EC)
  (logical : Omega → LogicalOrder (B2V3RecoveryRecord (Fin E) Fit Norm Frozen))
  (CAttr : Set Omega) (ledger : B2V3ActualPrimitiveLedger Omega T)
  {witnessSlot : B2V3ActiveSlot}
  {Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop}
  {maxSteps : Nat} {referenceAB referenceBA : Fit} {rThetaAB rThetaBA : ℝ}
  (binding : B2V3ActualSourceBindingFor witnessSlot ctx CAttr T ledger Key Owner Legal
    RejectReason maxSteps referenceAB referenceBA rThetaAB rThetaBA)
  (hfields : ∀ ω c, c ∈ ctx.guardedPool ω → SubmittedFields (logical ω) c)

include binding hfields

theorem source_current_recovery (u : ℝ) {ω : Omega}
    (ha : ω ∈ CAttr) (hr : ledger.rawGood ω)
    (hd : ω ∈ uniformDeviationEvent (currentPool (fun w => primaryOrder (logical w))) ctx.riskSpec u)
    (hg : ω ∈ selectorStrictPopulationGapEvent (currentPool (fun w => primaryOrder (logical w)))
      ctx.riskSpec ctx.truth u 0 0 0) :
    ω ∈ exactSelectedEvent (currentOutput (fun w => primaryOrder (logical w)) ctx.riskSpec) ctx.truth := by
  have hs := binding.exactSurvival_of_rawGood ha hr
  have hm := survival_mono ctx.guardedPool (currentPool (fun w => primaryOrder (logical w)))
    ctx.truth (fun w => sourcePool_subset_current _ _ (hfields w)) hs
  exact ordered_recovery _ ctx.riskSpec ctx.truth u ⟨⟨hm, hd⟩, hg⟩

theorem source_current_failure_bound
    (budgets : B2V3ActualRawTailBudgets ctx.jointLaw CAttr ledger)
    (u : ℝ) (δa δd δg : ENNReal)
    (ha : ctx.jointLaw CAttrᶜ ≤ δa)
    (hd : ctx.jointLaw (uniformDeviationEvent
      (currentPool (fun w => primaryOrder (logical w))) ctx.riskSpec u)ᶜ ≤ δd)
    (hg : ctx.jointLaw (selectorStrictPopulationGapEvent
      (currentPool (fun w => primaryOrder (logical w))) ctx.riskSpec ctx.truth u 0 0 0)ᶜ ≤ δg) :
    ctx.jointLaw (exactSelectedEvent
      (currentOutput (fun w => primaryOrder (logical w)) ctx.riskSpec) ctx.truth)ᶜ ≤
      δa + budgets.deltaR + δd + δg := by
  apply failure_bound ctx.jointLaw _ ctx.riskSpec ctx.truth u ctx.guardedPool CAttr
    (fun w => sourcePool_subset_current _ _ (hfields w)) δa budgets.deltaR δd δg ha
  · exact measure_b2V3ActualSourceFailure_le_deltaR_of_sourceBindingFor ctx CAttr ledger budgets binding
  · exact hd
  · exact hg
end Source
end
end BinaryFormal.B1Current

#print axioms BinaryFormal.B1Current.LogicalOrder
#print axioms BinaryFormal.B1Current.SubmittedFields
#print axioms BinaryFormal.B1Current.submittedFields_mem
#print axioms BinaryFormal.B1Current.sourcePool_subset_current
#print axioms BinaryFormal.B1Current.primaryRiskPair
#print axioms BinaryFormal.B1Current.primaryRiskPair_values
#print axioms BinaryFormal.B1Current.primaryRisk_values
#print axioms BinaryFormal.B1Current.currentPool
#print axioms BinaryFormal.B1Current.currentOutput
#print axioms BinaryFormal.B1Current.ordered_recovery
#print axioms BinaryFormal.B1Current.survival_mono
#print axioms BinaryFormal.B1Current.failure_subset
#print axioms BinaryFormal.B1Current.failure_bound
#print axioms BinaryFormal.B1Current.frozenOutput
#print axioms BinaryFormal.B1Current.exact_same_cluster
#print axioms BinaryFormal.B1Current.frozen_selected_exact
#print axioms BinaryFormal.B1Current.frozen_abstain
#print axioms BinaryFormal.B1Current.usable_exact_rows
#print axioms BinaryFormal.B1Current.source_current_recovery
#print axioms BinaryFormal.B1Current.source_current_failure_bound
