import BiostatFormal.HighDimensional.Multiview.Guards
import BiostatFormal.HighDimensional.Multiview.Risks
import BiostatFormal.HighDimensional.Multiview.RiskSelection

/-!
# Minimal multiview selector contract

This module records pointwise totality and empirical near-optimality for a selector over one
realized guarded pool.  It deliberately separates global abstention from risk inputs and does not
infer measurability or probability statements from totality.
-/

namespace BiostatFormal.HighDimensional.Multiview

universe uOmega uCandidate uEvaluationContext uCenter uLabel uSlot uRecordContext

/-- Reasons for a global selector abstention, distinct from per-candidate rejection. -/
inductive SelectorAbstainReason where
  | emptyPool
  | numericalFailure
  deriving DecidableEq, Repr

/-- One evaluation context and one shared population/empirical risk pair for a selector. -/
structure SelectorRiskSpec
    (Candidate : Type uCandidate) [RiskInput Candidate]
    (EvaluationContext : Type uEvaluationContext) where
  context : EvaluationContext
  risk : CompositeRiskPair Candidate

/--
Pointwise totality and empirical near-optimality over a realized guarded pool.  The risk
specification is retained as a parameter so later layers cannot silently mix evaluation contexts.
-/
structure SelectorContract
    {Omega : Type uOmega} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    (pool : Omega -> GuardedCandidatePool Candidate)
    (spec : SelectorRiskSpec Candidate EvaluationContext)
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (etaTie etaGuard etaNum : Real) : Prop where
  selects_of_nonempty :
    forall omega, (pool omega).Nonempty ->
      exists selected, output omega = .selected selected
  nearOptimal_of_selected :
    forall omega selected, output omega = .selected selected ->
      forall candidate : GuardedPoolMember (pool omega),
        empiricalCompositeRisk spec.risk selected.1 <=
          empiricalCompositeRisk spec.risk candidate.1 +
            selectionTolerance etaTie etaGuard etaNum

/-- The event that at least one exact contextual candidate survives the guards. -/
def exactSurvivalEvent
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    (pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext))
    (truth : Partition Center Label) : Set Omega :=
  {omega | exists z : GuardedPoolMember (pool omega),
    z.1 ∈ exactCandidateSet truth}

/-- Uniform empirical-to-population deviation over the realized guarded pool only. -/
def uniformDeviationEvent
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    (pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext))
    (spec : SelectorRiskSpec
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext)
    (u : Real) : Set Omega :=
  {omega | forall z : GuardedPoolMember (pool omega),
    |empiricalCompositeRisk spec.risk z.1 -
      populationCompositeRisk spec.risk z.1| <= u}

/--
Strict population separation of an exact survivor from every bad guarded candidate.  Outside the
exact-survival event the condition holds by the approved off-survival convention.
-/
def selectorStrictPopulationGapEvent
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    (pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext))
    (spec : SelectorRiskSpec
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext)
    (truth : Partition Center Label)
    (u etaTie etaGuard etaNum : Real) : Set Omega :=
  (exactSurvivalEvent pool truth)ᶜ ∪
    {omega | exists zGood : GuardedPoolMember (pool omega),
      zGood.1 ∈ exactCandidateSet truth /\
      forall z : GuardedPoolMember (pool omega),
        z.1 ∈ badCandidateSet truth ->
          populationCompositeRisk spec.risk zGood.1 + 2 * u +
              selectionTolerance etaTie etaGuard etaNum <
            populationCompositeRisk spec.risk z.1}

/-- A selected output is empirically near-optimal over its realized guarded pool. -/
def selectedNearOptimalEvent
    {Omega : Type uOmega} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    (pool : Omega -> GuardedCandidatePool Candidate)
    (spec : SelectorRiskSpec Candidate EvaluationContext)
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (etaTie etaGuard etaNum : Real) : Set Omega :=
  {omega | exists selected,
    output omega = .selected selected /\
    forall candidate : GuardedPoolMember (pool omega),
      empiricalCompositeRisk spec.risk selected.1 <=
        empiricalCompositeRisk spec.risk candidate.1 +
          selectionTolerance etaTie etaGuard etaNum}

/--
Exact survival together with selected near-optimality, extended off the survival event by union
with its complement.
-/
def selectorSuccessEvent
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    (pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext))
    (spec : SelectorRiskSpec
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext)
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label)
    (etaTie etaGuard etaNum : Real) : Set Omega :=
  (exactSurvivalEvent pool truth)ᶜ ∪
    selectedNearOptimalEvent pool spec output etaTie etaGuard etaNum

/-- An exact guarded witness makes the realized guarded pool nonempty. -/
theorem exactSurvivalEvent_nonempty
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    [Fintype Center] [DecidableEq Center]
    [Fintype Label] [DecidableEq Label]
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    (pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext))
    (truth : Partition Center Label) {omega : Omega} :
    omega ∈ exactSurvivalEvent pool truth -> (pool omega).Nonempty := by
  rintro ⟨z, _⟩
  exact ⟨z.1, z.2⟩

/-- An exact survivor forces a selected output under selector totality. -/
theorem selected_of_exactSurvival
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    {EvaluationContext : Type uEvaluationContext}
    [Fintype Center] [DecidableEq Center]
    [Fintype Label] [DecidableEq Label]
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    (pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext))
    (spec : SelectorRiskSpec
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext)
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label) (etaTie etaGuard etaNum : Real)
    (contract : SelectorContract pool spec output etaTie etaGuard etaNum)
    {omega : Omega} :
    omega ∈ exactSurvivalEvent pool truth ->
      exists selected, output omega = .selected selected := by
  intro hSurvival
  apply contract.selects_of_nonempty omega
  exact exactSurvivalEvent_nonempty pool truth hSurvival

/-- An exact survivor rules out every global abstention reason under selector totality. -/
theorem not_abstain_of_exactSurvival
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    {EvaluationContext : Type uEvaluationContext}
    [Fintype Center] [DecidableEq Center]
    [Fintype Label] [DecidableEq Label]
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    (pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext))
    (spec : SelectorRiskSpec
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext)
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label) (etaTie etaGuard etaNum : Real)
    (contract : SelectorContract pool spec output etaTie etaGuard etaNum)
    {omega : Omega} :
    omega ∈ exactSurvivalEvent pool truth ->
      forall reason, output omega ≠ .abstain reason := by
  intro hSurvival reason hAbstain
  obtain ⟨selected, hSelected⟩ := contract.selects_of_nonempty omega
    (exactSurvivalEvent_nonempty pool truth hSurvival)
  rw [hSelected] at hAbstain
  cases hAbstain

/-- Realizations where the output selects a globally-permutation-exact candidate. -/
def exactSelectedEvent
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    {pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext)}
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label) : Set Omega :=
  {omega | exists selected,
    output omega = .selected selected /\
      selected.1 ∈ exactCandidateSet truth}

/--
On exact survival, uniform guarded-pool deviation, a strict population gap, and selected
near-optimality imply exact selected output.
-/
theorem recoverySemanticCore
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    {EvaluationContext : Type uEvaluationContext}
    [Fintype Center] [DecidableEq Center]
    [Fintype Label] [DecidableEq Label]
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    (pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext))
    (spec : SelectorRiskSpec
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext)
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label)
    (u etaTie etaGuard etaNum : Real) :
    exactSurvivalEvent pool truth ∩
        uniformDeviationEvent pool spec u ∩
        selectorStrictPopulationGapEvent pool spec truth u etaTie etaGuard etaNum ∩
        selectorSuccessEvent pool spec output truth etaTie etaGuard etaNum ⊆
      exactSelectedEvent output truth := by
  rintro omega ⟨⟨⟨hSurvival, hDeviation⟩, hGapEvent⟩, hSuccessEvent⟩
  rcases hGapEvent with hOffSurvival | ⟨zGood, hGood, hGap⟩
  · exact (hOffSurvival hSurvival).elim
  rcases hSuccessEvent with hOffSurvival | ⟨zSelected, hOutput, hNearOptimal⟩
  · exact (hOffSurvival hSurvival).elim
  refine ⟨zSelected, hOutput, ?_⟩
  exact exact_of_guarded_risk_selection
    (pool omega) spec.risk truth zGood zSelected
    u etaTie etaGuard etaNum hGood hDeviation hGap (hNearOptimal zGood)

/-- An arbitrary extra coverage event may be conjoined and then discarded monotonically. -/
theorem recoverySemanticWithCoverage
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    {EvaluationContext : Type uEvaluationContext}
    [Fintype Center] [DecidableEq Center]
    [Fintype Label] [DecidableEq Label]
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    (pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext))
    (spec : SelectorRiskSpec
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext)
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label)
    (u etaTie etaGuard etaNum : Real)
    (coverageEvent : Set Omega) :
    coverageEvent ∩ exactSurvivalEvent pool truth ∩
        uniformDeviationEvent pool spec u ∩
        selectorStrictPopulationGapEvent pool spec truth u etaTie etaGuard etaNum ∩
        selectorSuccessEvent pool spec output truth etaTie etaGuard etaNum ⊆
      exactSelectedEvent output truth := by
  rintro omega ⟨⟨⟨⟨_, hSurvival⟩, hDeviation⟩, hGapEvent⟩, hSuccessEvent⟩
  exact recoverySemanticCore pool spec output truth u etaTie etaGuard etaNum
    ⟨⟨⟨hSurvival, hDeviation⟩, hGapEvent⟩, hSuccessEvent⟩

/-- Realizations where the output selects a complete but non-exact candidate. -/
def wrongSelectedEvent
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    {pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext)}
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label) : Set Omega :=
  {omega | exists selected,
    output omega = .selected selected /\
      selected.1 ∈ badCandidateSet truth}

/-- Realizations where the selector explicitly abstains for a global reason. -/
def abstainEvent
    {Omega : Type uOmega} {Candidate : Type uCandidate} [DecidableEq Candidate]
    {pool : Omega -> GuardedCandidatePool Candidate}
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason) : Set Omega :=
  {omega | exists reason, output omega = .abstain reason}

/-- Exact and wrong selected states are disjoint. -/
theorem exactSelected_wrongSelected_disjoint
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    {pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext)}
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label) :
    Disjoint (exactSelectedEvent output truth) (wrongSelectedEvent output truth) := by
  rw [Set.disjoint_left]
  rintro omega ⟨exactSelected, hExactOutput, hExact⟩
    ⟨wrongSelected, hWrongOutput, hWrong⟩
  rw [hExactOutput] at hWrongOutput
  cases hWrongOutput
  have hNotExact : ¬ Exact exactSelected.1.label truth := by
    simpa [badCandidateSet] using hWrong
  exact hNotExact (by simpa [exactCandidateSet] using hExact)

/-- Exact selected states and abstentions are disjoint output constructors. -/
theorem exactSelected_abstain_disjoint
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    {pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext)}
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label) :
    Disjoint (exactSelectedEvent output truth) (abstainEvent output) := by
  rw [Set.disjoint_left]
  rintro omega ⟨selected, hSelected, -⟩ ⟨reason, hAbstain⟩
  rw [hSelected] at hAbstain
  cases hAbstain

/-- Wrong selected states and abstentions are disjoint output constructors. -/
theorem wrongSelected_abstain_disjoint
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    {pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext)}
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label) :
    Disjoint (wrongSelectedEvent output truth) (abstainEvent output) := by
  rw [Set.disjoint_left]
  rintro omega ⟨selected, hSelected, -⟩ ⟨reason, hAbstain⟩
  rw [hSelected] at hAbstain
  cases hAbstain

/-- Every output is exactly selected, wrongly selected, or a global abstention. -/
theorem exactWrongAbstain_union_eq_univ
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    {pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext)}
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label) :
    exactSelectedEvent output truth ∪
        (wrongSelectedEvent output truth ∪ abstainEvent output) = Set.univ := by
  classical
  ext omega
  constructor
  · intro _
    simp
  · intro _
    cases hOutput : output omega with
    | selected selected =>
        by_cases hExact : selected.1 ∈ exactCandidateSet truth
        · exact Or.inl ⟨selected, hOutput, hExact⟩
        · right
          left
          refine ⟨selected, hOutput, ?_⟩
          simpa [badCandidateSet, exactCandidateSet] using hExact
    | abstain reason =>
        right
        right
        exact ⟨reason, hOutput⟩

/-- Failure of exact selection splits into a wrong selection or a global abstention. -/
theorem exactSelected_compl_eq_wrongSelected_union_abstain
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    {pool : Omega -> GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext)}
    (output : (omega : Omega) ->
      AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label) :
    (exactSelectedEvent output truth)ᶜ =
      wrongSelectedEvent output truth ∪ abstainEvent output := by
  classical
  ext omega
  constructor
  · intro hNotExact
    cases hOutput : output omega with
    | selected selected =>
        left
        refine ⟨selected, hOutput, ?_⟩
        by_contra hBad
        apply hNotExact
        exact ⟨selected, hOutput, by simpa [badCandidateSet, exactCandidateSet] using hBad⟩
    | abstain reason =>
        right
        exact ⟨reason, hOutput⟩
  · rintro (hWrong | hAbstain) hExact
    · exact Set.disjoint_left.mp
        (exactSelected_wrongSelected_disjoint output truth) hExact hWrong
    · exact Set.disjoint_left.mp
        (exactSelected_abstain_disjoint output truth) hExact hAbstain

end BiostatFormal.HighDimensional.Multiview
