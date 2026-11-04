import BiostatFormal.HighDimensional.Multiview.A1WitnessTarget
import BiostatFormal.HighDimensional.Multiview.A3Terminal
import BiostatFormal.HighDimensional.Multiview.B2V3RecoveryContext
import BiostatFormal.HighDimensional.Multiview.B2V3PseudoTrueRisk
import BiostatFormal.HighDimensional.Multiview.CanonicalSelector
import BiostatFormal.HighDimensional.Multiview.SelectorMeasurability
import Mathlib.Tactic.Linarith

/-!
# B2-v3 deterministic connections

This module connects a preregistered ordinary-KMeans A1 witness and the local A3 terminal
guarded pool to the B2-v3 recovery context.  It contains only pointwise deterministic transport:
no probability theorem or source deduplication implementation is introduced here.
-/

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

universe uOmega uCenter uFit uNorm uFrozen uEval uReject uFailure

/-- The A1 interface for the B2-v3 active-slot type.  Every inhabitant is algorithmically active. -/
def b2V3ActiveSlotInterface : A1SlotInterface B2V3ActiveSlot where
  view := B2V3ActiveSlot.view
  status _ := .active

/-- An ordinary-KMeans slot is a view-backed active A1 witness under the v3 configuration. -/
theorem b2V3OrdinaryKMeansCoverageSlot_isActiveWitness
    (w : B2V3ActiveSlot)
    (hW : w ∈ b2V3OrdinaryKMeansCoverageSlotOrder) :
    b2V3ActiveSlotInterface.IsActiveWitnessSlot b2V3ViewConfiguration w := by
  cases w <;>
    simp [A1SlotInterface.IsActiveWitnessSlot, b2V3ActiveSlotInterface,
      b2V3ViewConfiguration, B2V3ActiveSlot.view,
      b2V3OrdinaryKMeansCoverageSlotOrder] at hW ⊢

/--
A fixed ordinary-KMeans witness chosen before observing the realization injects into generic
B2-v3 A1 exact coverage.  The witness is a theorem parameter, never a realization-dependent map.
-/
theorem b2V3FixedSlotCoverageEvent_subset
    {Omega : Type uOmega}
    (splitBalance : Set Omega)
    (pairExact : B2V3ActiveSlot → Set Omega)
    (w : B2V3ActiveSlot)
    (hW : w ∈ b2V3OrdinaryKMeansCoverageSlotOrder) :
    a1FixedSlotCoverageEvent splitBalance pairExact w ⊆
      a1ExactCoverageEvent splitBalance b2V3ViewConfiguration b2V3ActiveSlotInterface
        b2V3ActiveSlotOrder.toFinset pairExact := by
  apply a1FixedSlotCoverageEvent_subset splitBalance b2V3ViewConfiguration
    b2V3ActiveSlotInterface b2V3ActiveSlotOrder.toFinset pairExact w
  · simp only [List.mem_toFinset]
    cases w <;>
      simp [b2V3OrdinaryKMeansCoverageSlotOrder, b2V3ActiveSlotOrder] at hW ⊢
  · exact b2V3OrdinaryKMeansCoverageSlot_isActiveWitness w hW

/--
Source-conformance data for one local A3 guarded pool through the raw/global union, a canonical
representative, canonical selector input, and the recovery context's guarded pool.  The structure
records source facts; it deliberately does not implement a new deduplication algorithm.
-/
structure B2V3LocalToGlobalTrace
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm}
    {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (omega : Omega)
    {Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop}
    (localA3Pool : GuardedCandidatePool
      (RiskCandidate (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal)) where
  /-- The raw/global union before canonical representative selection. -/
  rawGlobalPool : GuardedCandidatePool (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)
  /-- Erasing only the legality proof carries each local A3 survivor into the raw/global union. -/
  local_to_raw : ∀ candidate, candidate ∈ localA3Pool → candidate.1 ∈ rawGlobalPool
  /-- A representative retained for each local A3 survivor. -/
  canonicalRepresentative :
    RiskCandidate (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal →
      B2V3RecoveryRecord (Fin E) Fit Norm Frozen
  /-- The retained representative belongs to the raw/global union. -/
  representative_mem_rawGlobal : ∀ candidate, candidate.1 ∈ rawGlobalPool →
    canonicalRepresentative candidate ∈ rawGlobalPool
  /-- Same label preserves exactness under the existing permutation-invariant `Exact` predicate. -/
  representative_label_eq : ∀ candidate,
    (canonicalRepresentative candidate).label = candidate.1.label
  /-- Named raw canonical records before partition-level deduplication. -/
  rawCanonicalRecords : List (CanonicalCandidateRecord E
    (B2V3ActiveSlot × Direction × CandidateStage)
    (B2V3RecoveryRecord (Fin E) Fit Norm Frozen))
  rawGlobalPool_eq_rawCanonicalRecordPayloads : rawGlobalPool =
    (rawCanonicalRecords.map CanonicalCandidateRecord.payload).toFinset
  rawCanonicalRecord_payload_mem_rawGlobal : ∀ record, record ∈ rawCanonicalRecords →
    record.payload ∈ rawGlobalPool
  /-- The frozen canonical selector input supplied by the source-facing pipeline. -/
  canonicalInput : CanonicalSelectorInput E
    (B2V3ActiveSlot × Direction × CandidateStage)
    (B2V3RecoveryRecord (Fin E) Fit Norm Frozen) EvaluationContext ctx.riskSpec
  /-- The canonical input record associated with each local A3 survivor. -/
  canonicalRecord :
    RiskCandidate (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal →
      CanonicalCandidateRecord E (B2V3ActiveSlot × Direction × CandidateStage)
        (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)
  /-- The canonical record has the key of its representative's partition. -/
  canonicalRecord_partitionKey : ∀ candidate,
    (canonicalRecord candidate).partitionKey =
      canonicalPartitionKey (canonicalRepresentative candidate).label
  /-- All raw source provenance for that canonical key is retained in the canonical record. -/
  canonicalRecord_provenance_eq : ∀ candidate,
    (canonicalRecord candidate).provenance =
      provenanceForPartitionKey rawCanonicalRecords (canonicalRecord candidate).partitionKey
  /-- Canonical deduplication leaves no repeated canonical partition keys. -/
  canonicalInput_partitionKey_nodup :
    (canonicalInput.records.map CanonicalCandidateRecord.partitionKey).Nodup
  /-- A raw representative contributes an actual frozen canonical-selector input record. -/
  canonicalRecord_mem_input : ∀ candidate,
    canonicalRepresentative candidate ∈ rawGlobalPool →
    canonicalRecord candidate ∈ canonicalInput.records
  canonicalRecord_payload_eq : ∀ candidate,
    (canonicalRecord candidate).payload = canonicalRepresentative candidate
  /-- The post-dedup canonical payload pool is exactly the recovery context's guarded pool. -/
  canonicalInput_pool_eq_guardedPool : canonicalInput.pool = ctx.guardedPool omega

namespace B2V3LocalToGlobalTrace

variable {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
  {Fit : Type uFit} {Norm : Type uNorm}
  {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
  [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
  {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}
  {omega : Omega}
  {Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop}
  {localA3Pool : GuardedCandidatePool
    (RiskCandidate (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal)}

/-- A local exact A3 survivor has an exact canonical representative. -/
theorem representative_exact_of_local_exact
    (trace : B2V3LocalToGlobalTrace ctx omega localA3Pool)
    (candidate : RiskCandidate (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal)
    (hExact : Exact candidate.1.label ctx.truth) :
    Exact (trace.canonicalRepresentative candidate).label ctx.truth := by
  rw [trace.representative_label_eq]
  exact hExact

/-- The trace transports one local A3 survivor into `ctx.guardedPool omega`. -/
theorem representative_mem_guardedPool_of_local_mem
    (trace : B2V3LocalToGlobalTrace ctx omega localA3Pool)
    (candidate : RiskCandidate (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal)
    (hLocal : candidate ∈ localA3Pool) :
    trace.canonicalRepresentative candidate ∈ ctx.guardedPool omega := by
  have hLocalRaw : candidate.1 ∈ trace.rawGlobalPool :=
    trace.local_to_raw candidate hLocal
  have hRepresentativeRaw : trace.canonicalRepresentative candidate ∈ trace.rawGlobalPool :=
    trace.representative_mem_rawGlobal candidate hLocalRaw
  have hRecord : trace.canonicalRecord candidate ∈ trace.canonicalInput.records :=
    trace.canonicalRecord_mem_input candidate hRepresentativeRaw
  have hInput : (trace.canonicalRecord candidate).payload ∈ trace.canonicalInput.pool :=
    trace.canonicalInput.payload_mem_pool hRecord
  have hRepresentativeInput : trace.canonicalRepresentative candidate ∈ trace.canonicalInput.pool := by
    simpa [trace.canonicalRecord_payload_eq candidate] using hInput
  rw [← trace.canonicalInput_pool_eq_guardedPool]
  exact hRepresentativeInput

/-- A local exact A3 survivor yields B2-v3 global exact survival. -/
theorem exactSurvival
    (trace : B2V3LocalToGlobalTrace ctx omega localA3Pool)
    (candidate : RiskCandidate (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal)
    (hLocal : candidate ∈ localA3Pool)
    (hExact : Exact candidate.1.label ctx.truth) :
    omega ∈ ctx.exactSurvival := by
  refine ⟨⟨trace.canonicalRepresentative candidate,
    trace.representative_mem_guardedPool_of_local_mem candidate hLocal⟩, ?_⟩
  exact trace.representative_exact_of_local_exact candidate hExact

end B2V3LocalToGlobalTrace

/--
The A3 terminal certificate supplies an exact local survivor; the conformance trace transports it
to the B2-v3 recovery context without erasing legality except through `.1`.
-/
theorem b2V3ExactSurvival_of_terminalOutcome
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} [SeminormedAddCommGroup Fit]
    {Norm : Type uNorm} {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (omega : Omega)
    {Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop}
    {assignedFromAB assignedFromBA : Fit → Partition (Fin E) (Fin 3) → Prop}
    {FailureReason : Type uFailure} {RejectReason : Type uReject}
    {slot : B2V3ActiveSlot} {maxSteps : Nat}
    (postOK : PairedRiskCandidates (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
      slot .post → Prop)
    (preOK : PairedRiskCandidates (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
      slot .pre → Prop)
    (rejectReason : RejectReason)
    (prePair : PairedRiskCandidates (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
      slot .pre)
    (outcome : PairedTerminalOutcome (Fin E) Fit B2V3ActiveSlot Norm Frozen Legal ctx.truth
      assignedFromAB assignedFromBA FailureReason slot maxSteps)
    (trace : B2V3LocalToGlobalTrace ctx omega
      (atomicGuard postOK preOK rejectReason
        (prePair.toPairedCandidateState :
          PairedCandidateState (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
            FailureReason slot .pre)
        outcome.toPairedCandidateState).toGuardedPool)
    (hPreExact : prePair.Exact ctx.truth)
    (hPreOK : preOK prePair) :
    omega ∈ ctx.exactSurvival := by
  obtain ⟨candidate, hLocal, hExact⟩ :=
    outcome.exists_exact_mem_guardedPool postOK preOK rejectReason prePair hPreExact hPreOK
  exact trace.exactSurvival candidate hLocal hExact

/--
For the same preregistered slot, fixed-slot A1 exact coverage and terminal A3 transport hold
pointwise together.  No witness is selected after observing `omega`.
-/
theorem b2V3FixedSlotCoverage_and_exactSurvival
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} [SeminormedAddCommGroup Fit]
    {Norm : Type uNorm} {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (splitBalance : Set Omega)
    (pairExact : B2V3ActiveSlot → Set Omega)
    (w : B2V3ActiveSlot)
    (hW : w ∈ b2V3OrdinaryKMeansCoverageSlotOrder)
    (omega : Omega)
    (hFixedCoverage : omega ∈ a1FixedSlotCoverageEvent splitBalance pairExact w)
    {Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop}
    {assignedFromAB assignedFromBA : Fit → Partition (Fin E) (Fin 3) → Prop}
    {FailureReason : Type uFailure} {RejectReason : Type uReject}
    {maxSteps : Nat}
    (postOK : PairedRiskCandidates (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
      w .post → Prop)
    (preOK : PairedRiskCandidates (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
      w .pre → Prop)
    (rejectReason : RejectReason)
    (prePair : PairedRiskCandidates (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
      w .pre)
    (outcome : PairedTerminalOutcome (Fin E) Fit B2V3ActiveSlot Norm Frozen Legal ctx.truth
      assignedFromAB assignedFromBA FailureReason w maxSteps)
    (trace : B2V3LocalToGlobalTrace ctx omega
      (atomicGuard postOK preOK rejectReason
        (prePair.toPairedCandidateState :
          PairedCandidateState (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
            FailureReason w .pre)
        outcome.toPairedCandidateState).toGuardedPool)
    (fixedPairExact_to_preExact : omega ∈ pairExact w → prePair.Exact ctx.truth)
    (hPreOK : preOK prePair) :
    omega ∈ a1ExactCoverageEvent splitBalance b2V3ViewConfiguration b2V3ActiveSlotInterface
        b2V3ActiveSlotOrder.toFinset pairExact ∧
      omega ∈ ctx.exactSurvival := by
  constructor
  · exact b2V3FixedSlotCoverageEvent_subset splitBalance pairExact w hW hFixedCoverage
  · exact b2V3ExactSurvival_of_terminalOutcome ctx omega postOK preOK rejectReason prePair
      outcome trace (fixedPairExact_to_preExact hFixedCoverage.2) hPreOK

/-! ## Connections 3--5: frozen envelope, slope transport, and source measurability -/

/-- The fixed, preregistered sixty-record B2-v3 envelope. -/
abbrev B2V3EnvelopeIndex := { index // index ∈ b2V3CandidateRecordEnvelope }

/--
Source-conformance bridge from the frozen sixty-index envelope to the realized post-dedup pool.
An unrealized index carries `none`, so rejected runs never require a fabricated risk input.
-/
structure B2V3EnvelopePayloadBridge
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext) where
  primaryIndex : ∀ {omega : Omega},
    GuardedPoolMember (ctx.guardedPool omega) → B2V3EnvelopeIndex
  realizedPayload : Omega → B2V3EnvelopeIndex → Option (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)
  primaryIndex_realized : ∀ (omega : Omega)
    (payload : GuardedPoolMember (ctx.guardedPool omega)),
    realizedPayload omega (primaryIndex payload) = some payload.1
  primaryIndex_unique : ∀ (omega : Omega)
    (payload : GuardedPoolMember (ctx.guardedPool omega)) (index : B2V3EnvelopeIndex),
    realizedPayload omega index = some payload.1 → index = primaryIndex payload
  /-- Frozen A2 empirical risk at every preregistered index. -/
  empiricalAt : B2V3EnvelopeIndex → Omega → ℝ
  /-- Frozen A2 population risk at every preregistered index. -/
  populationAt : B2V3EnvelopeIndex → ℝ
  primaryIndex_empiricalRisk : ∀ (omega : Omega)
    (payload : GuardedPoolMember (ctx.guardedPool omega)),
    empiricalAt (primaryIndex payload) omega =
      empiricalCompositeRisk ctx.riskSpec.risk payload.1
  primaryIndex_populationRisk : ∀ (omega : Omega)
    (payload : GuardedPoolMember (ctx.guardedPool omega)),
    populationAt (primaryIndex payload) =
      populationCompositeRisk ctx.riskSpec.risk payload.1

namespace B2V3EnvelopePayloadBridge

variable {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
  {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
  {EvaluationContext : Type uEval}
  [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
  {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}

/-- Each actual post-dedup payload has exactly one designated frozen primary index. -/
theorem exists_unique_primaryIndex
    (bridge : B2V3EnvelopePayloadBridge ctx) (omega : Omega)
    (payload : GuardedPoolMember (ctx.guardedPool omega)) :
    ∃! index : B2V3EnvelopeIndex,
      bridge.realizedPayload omega index = some payload.1 := by
  refine ⟨bridge.primaryIndex payload, bridge.primaryIndex_realized omega payload, ?_⟩
  intro index hIndex
  exact bridge.primaryIndex_unique omega payload index hIndex

end B2V3EnvelopePayloadBridge

/-- Good deviation over every index of the frozen A2 sixty-record envelope. -/
def b2V3EnvelopeGoodDeviation
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}
    (bridge : B2V3EnvelopePayloadBridge ctx) (u : ℝ) : Set Omega :=
  {omega | ∀ index, |bridge.empiricalAt index omega - bridge.populationAt index| ≤ u}

/-- Frozen envelope control transports to the actual guarded-pool deviation event. -/
theorem b2V3EnvelopeGoodDeviation_subset_uniformDeviation
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}
    (bridge : B2V3EnvelopePayloadBridge ctx) (u : ℝ) :
    b2V3EnvelopeGoodDeviation bridge u ⊆ ctx.uniformDeviation u := by
  intro omega hGood payload
  have hFrozen := hGood (bridge.primaryIndex payload)
  rw [bridge.primaryIndex_empiricalRisk omega payload,
    bridge.primaryIndex_populationRisk omega payload] at hFrozen
  exact hFrozen

/-- A nonnegative recordwise slope and a one-error margin give the selector's strict gap. -/
theorem strictPopulationGap_of_slope
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (omega : Omega) (zGood : GuardedPoolMember (ctx.guardedPool omega))
    (kappa u etaTie etaGuard etaNum : ℝ)
    (hGood : zGood.1 ∈ exactCandidateSet ctx.truth)
    (hKappa : 0 ≤ kappa)
    (hE : 0 < (E : ℝ))
    (hSlope : ∀ z : GuardedPoolMember (ctx.guardedPool omega),
      z.1 ∈ badCandidateSet ctx.truth →
      kappa * labelDistance z.1.label ctx.truth ≤
        populationCompositeRisk ctx.riskSpec.risk z.1 -
          populationCompositeRisk ctx.riskSpec.risk zGood.1)
    (hMargin : 2 * u + selectionTolerance etaTie etaGuard etaNum < kappa / E) :
    omega ∈ ctx.strictPopulationGap u etaTie etaGuard etaNum := by
  right
  refine ⟨zGood, hGood, ?_⟩
  intro z hBad
  have hENat : 0 < E := by exact_mod_cast hE
  letI : Nonempty (Fin E) := ⟨⟨0, hENat⟩⟩
  have hDistance : 1 / (E : ℝ) ≤ labelDistance z.1.label ctx.truth := by
    simpa using one_div_card_le_labelDistance_of_bad (candidate := z.1.label)
      (truth := ctx.truth) hBad
  have hScaled : kappa / (E : ℝ) ≤ kappa * labelDistance z.1.label ctx.truth := by
    calc
      kappa / (E : ℝ) = kappa * (1 / (E : ℝ)) := by ring
      _ ≤ kappa * labelDistance z.1.label ctx.truth :=
        mul_le_mul_of_nonneg_left hDistance hKappa
  have hRisk := hSlope z hBad
  linarith

/--
B2-v3 specialization: the sole slope is the net compatibility slope, and population-risk
coherence remains an explicit source premise in `hSlope`.
-/
theorem b2V3StrictPopulationGap_of_netSlope
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (omega : Omega) (zGood : GuardedPoolMember (ctx.guardedPool omega))
    (gMin outcomeDefect treatmentDefect Jmin u etaTie etaGuard etaNum : ℝ)
    (hCenters : E = 9 ∨ E = 12)
    (hJmin : 0 < Jmin)
    (hDefect : b2V3FusionDefect outcomeDefect treatmentDefect < gMin * Jmin)
    (hGood : zGood.1 ∈ exactCandidateSet ctx.truth)
    (hSlope : ∀ z : GuardedPoolMember (ctx.guardedPool omega),
      z.1 ∈ badCandidateSet ctx.truth →
      b2V3NetCompatibilitySlope E gMin
          (b2V3FusionDefect outcomeDefect treatmentDefect) Jmin *
          labelDistance z.1.label ctx.truth ≤
        populationCompositeRisk ctx.riskSpec.risk z.1 -
          populationCompositeRisk ctx.riskSpec.risk zGood.1)
    (hMargin : 2 * u + selectionTolerance etaTie etaGuard etaNum <
      b2V3NetCompatibilitySlope E gMin
          (b2V3FusionDefect outcomeDefect treatmentDefect) Jmin / E) :
    omega ∈ ctx.strictPopulationGap u etaTie etaGuard etaNum := by
  have hKappaPos : 0 < b2V3NetCompatibilitySlope E gMin
      (b2V3FusionDefect outcomeDefect treatmentDefect) Jmin := by
    simpa using b2V3NetCompatibilitySlope_pos (Center := Fin E) (by simpa using hCenters)
      gMin (b2V3FusionDefect outcomeDefect treatmentDefect) Jmin hJmin hDefect
  have hE : 0 < (E : ℝ) := by
    rcases hCenters with hE | hE <;> subst E <;> norm_num
  exact strictPopulationGap_of_slope ctx omega zGood
    (b2V3NetCompatibilitySlope E gMin
      (b2V3FusionDefect outcomeDefect treatmentDefect) Jmin)
    u etaTie etaGuard etaNum hGood hKappaPos.le hE hSlope hMargin

/--
The source-facing fixed-index canonical-selector adapter.  The equality to the actual guarded
pool and all fixed-index pullbacks are explicit source-conformance fields.
-/
structure B2V3CanonicalSelectorSourceAdapter
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    [MeasurableSpace (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext) where
  envelope : FiniteCanonicalSelectorEnvelope Omega B2V3EnvelopeIndex E
    (B2V3ActiveSlot × Direction × CandidateStage)
    (B2V3RecoveryRecord (Fin E) Fit Norm Frozen) EvaluationContext ctx.riskSpec
  pool_eq : ∀ omega, (envelope.toInput omega).pool = ctx.guardedPool omega
  coordinates : CanonicalSelectorMeasurabilityAssumptions envelope
  actualOutput : (omega : Omega) → AlgorithmOutput
    (envelope.toInput omega).pool SelectorAbstainReason
  fixedIndex_selectedOutput_measurable : ∀ index, MeasurableSet {omega |
    (actualOutput omega).toSum = Sum.inr (envelope.record omega index).payload}
  fixedIndex_permutationExact_measurable : ∀ index, MeasurableSet {omega |
    (envelope.record omega index).payload ∈ exactCandidateSet ctx.truth}

namespace B2V3CanonicalSelectorSourceAdapter

variable {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
  {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
  {EvaluationContext : Type uEval}
  [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
  [MeasurableSpace (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
  {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}

/-- Erased canonical reference output, whose codomain no longer depends on the realized pool. -/
noncomputable def canonicalReferenceErasedOutput
    (source : B2V3CanonicalSelectorSourceAdapter ctx)
    (etaTie etaGuard etaNum : ℝ) : Omega →
      SelectorAbstainReason ⊕ B2V3RecoveryRecord (Fin E) Fit Norm Frozen :=
  fun omega => (canonicalOutput source.envelope.toInput etaTie etaGuard etaNum omega).toSum

/-- Transport the actual source output to the recovery context using its stated pool equality. -/
noncomputable def outputInContext
    (source : B2V3CanonicalSelectorSourceAdapter ctx) : ctx.SelectorOutput :=
  fun omega => cast
    (congrArg (fun pool => AlgorithmOutput pool SelectorAbstainReason) (source.pool_eq omega))
    (source.actualOutput omega)

private theorem algorithmOutput_toSum_cast
    {Candidate : Type*} [DecidableEq Candidate]
    {firstPool secondPool : GuardedCandidatePool Candidate}
    (hPool : firstPool = secondPool)
    (output : AlgorithmOutput firstPool SelectorAbstainReason) :
    (cast (congrArg (fun pool => AlgorithmOutput pool SelectorAbstainReason) hPool) output).toSum =
      output.toSum := by
  cases hPool
  rfl

/-- Transport along a pool equality preserves the erased output exactly. -/
theorem outputInContext_toSum
    (source : B2V3CanonicalSelectorSourceAdapter ctx) (omega : Omega) :
    (source.outputInContext omega).toSum = (source.actualOutput omega).toSum := by
  unfold outputInContext
  exact algorithmOutput_toSum_cast (source.pool_eq omega) (source.actualOutput omega)

/-- The erased context output is measurable whenever the source erased output is measurable. -/
theorem measurable_outputInContext_toSum
    (source : B2V3CanonicalSelectorSourceAdapter ctx)
    (hOutput : Measurable (fun omega => (source.actualOutput omega).toSum)) :
    Measurable (fun omega => (source.outputInContext omega).toSum) := by
  rw [show (fun omega => (source.outputInContext omega).toSum) =
    (fun omega => (source.actualOutput omega).toSum) by
      funext omega
      exact source.outputInContext_toSum omega]
  exact hOutput

/-- A selected-payload event expressed using the pool-independent erased output. -/
private def erasedSelectedPayloadEvent
    {Omega Candidate : Type*} (erased : Omega → SelectorAbstainReason ⊕ Candidate)
    (predicate : Candidate → Prop) : Set Omega :=
  {omega | ∃ candidate, erased omega = Sum.inr candidate ∧ predicate candidate}

/-- The dependent selected-output event has the stated erased characterization. -/
private lemma selectedPayloadEvent_eq_erased
    {Omega Candidate : Type*} [DecidableEq Candidate]
    {pool : Omega → GuardedCandidatePool Candidate}
    (output : (omega : Omega) → AlgorithmOutput (pool omega) SelectorAbstainReason)
    (predicate : Candidate → Prop) :
    {omega | ∃ selected, output omega = .selected selected ∧ predicate selected.1} =
      erasedSelectedPayloadEvent (fun omega => (output omega).toSum) predicate := by
  ext omega
  constructor
  · rintro ⟨selected, hSelected, hPredicate⟩
    exact ⟨selected.1, by simp [hSelected], hPredicate⟩
  · rintro ⟨candidate, hSum, hPredicate⟩
    generalize hOutput : output omega = state at hSum ⊢
    cases state with
    | abstain reason => simp [hOutput] at hSum
    | selected selected =>
        have hPayload : selected.1 = candidate := by simpa [hOutput] using hSum
        exact ⟨selected, hOutput, by simpa [hPayload] using hPredicate⟩

/-- An abstention event expressed using the pool-independent erased output. -/
private def erasedAbstainEvent
    {Omega Candidate : Type*} (erased : Omega → SelectorAbstainReason ⊕ Candidate) : Set Omega :=
  {omega | ∃ reason, erased omega = Sum.inl reason}

/-- The dependent abstention event has the stated erased characterization. -/
private lemma abstainEvent_eq_erased
    {Omega Candidate : Type*} [DecidableEq Candidate]
    {pool : Omega → GuardedCandidatePool Candidate}
    (output : (omega : Omega) → AlgorithmOutput (pool omega) SelectorAbstainReason) :
    abstainEvent output = erasedAbstainEvent (fun omega => (output omega).toSum) := by
  ext omega
  constructor
  · rintro ⟨reason, hOutput⟩
    exact ⟨reason, by simp [hOutput]⟩
  · rintro ⟨reason, hSum⟩
    generalize hOutput : output omega = state at hSum ⊢
    cases state with
    | abstain other => exact ⟨other, hOutput⟩
    | selected selected => simp [hOutput] at hSum

/-- The canonical reference erased output is measurable from the named coordinate families. -/
theorem measurable_canonicalReferenceErasedOutput
    (source : B2V3CanonicalSelectorSourceAdapter ctx)
    (etaTie etaGuard etaNum : ℝ) :
    Measurable (source.canonicalReferenceErasedOutput etaTie etaGuard etaNum) := by
  exact measurable_canonicalOutput_toSum source.envelope etaTie etaGuard etaNum source.coordinates

/-- The explicit selected-output and permutation-exactness pullbacks feed the existing theorem. -/
theorem measurableSet_actualExactSelected
    (source : B2V3CanonicalSelectorSourceAdapter ctx) :
    MeasurableSet (exactSelectedEvent source.actualOutput ctx.truth) := by
  exact measurableSet_exactSelectedEvent source.envelope source.actualOutput ctx.truth
    source.fixedIndex_selectedOutput_measurable source.fixedIndex_permutationExact_measurable

/-- The explicit selected-output and permutation-exactness pullbacks feed the existing theorem. -/
theorem measurableSet_actualWrongSelected
    (source : B2V3CanonicalSelectorSourceAdapter ctx)
    (hOutput : Measurable (fun omega => (source.actualOutput omega).toSum)) :
    MeasurableSet (wrongSelectedEvent source.actualOutput ctx.truth) := by
  exact measurableSet_wrongSelectedEvent source.envelope source.actualOutput ctx.truth hOutput
    source.fixedIndex_selectedOutput_measurable source.fixedIndex_permutationExact_measurable

/-- Exact selection is unchanged by transport through the stated pool equality. -/
theorem outputInContext_exactSelected_eq
    (source : B2V3CanonicalSelectorSourceAdapter ctx) :
    ctx.exactSelected source.outputInContext =
      exactSelectedEvent source.actualOutput ctx.truth := by
  change exactSelectedEvent source.outputInContext ctx.truth =
    exactSelectedEvent source.actualOutput ctx.truth
  calc
    exactSelectedEvent source.outputInContext ctx.truth =
        erasedSelectedPayloadEvent
          (fun omega => (source.outputInContext omega).toSum)
          (fun candidate => candidate ∈ exactCandidateSet ctx.truth) :=
      selectedPayloadEvent_eq_erased source.outputInContext _
    _ = erasedSelectedPayloadEvent
          (fun omega => (source.actualOutput omega).toSum)
          (fun candidate => candidate ∈ exactCandidateSet ctx.truth) := by
      rw [show (fun omega => (source.outputInContext omega).toSum) =
        (fun omega => (source.actualOutput omega).toSum) by
          funext omega
          exact source.outputInContext_toSum omega]
    _ = exactSelectedEvent source.actualOutput ctx.truth :=
      (selectedPayloadEvent_eq_erased source.actualOutput _).symm

/-- Wrong selection is unchanged by transport through the stated pool equality. -/
theorem outputInContext_wrongSelected_eq
    (source : B2V3CanonicalSelectorSourceAdapter ctx) :
    wrongSelectedEvent source.outputInContext ctx.truth =
      wrongSelectedEvent source.actualOutput ctx.truth := by
  calc
    wrongSelectedEvent source.outputInContext ctx.truth =
        erasedSelectedPayloadEvent
          (fun omega => (source.outputInContext omega).toSum)
          (fun candidate => candidate ∈ badCandidateSet ctx.truth) :=
      selectedPayloadEvent_eq_erased source.outputInContext _
    _ = erasedSelectedPayloadEvent
          (fun omega => (source.actualOutput omega).toSum)
          (fun candidate => candidate ∈ badCandidateSet ctx.truth) := by
      rw [show (fun omega => (source.outputInContext omega).toSum) =
        (fun omega => (source.actualOutput omega).toSum) by
          funext omega
          exact source.outputInContext_toSum omega]
    _ = wrongSelectedEvent source.actualOutput ctx.truth :=
      (selectedPayloadEvent_eq_erased source.actualOutput _).symm

/-- Abstention is unchanged by transport through the stated pool equality. -/
theorem outputInContext_abstain_eq
    (source : B2V3CanonicalSelectorSourceAdapter ctx) :
    abstainEvent source.outputInContext = abstainEvent source.actualOutput := by
  calc
    abstainEvent source.outputInContext =
        erasedAbstainEvent (fun omega => (source.outputInContext omega).toSum) :=
      abstainEvent_eq_erased source.outputInContext
    _ = erasedAbstainEvent (fun omega => (source.actualOutput omega).toSum) := by
      rw [show (fun omega => (source.outputInContext omega).toSum) =
        (fun omega => (source.actualOutput omega).toSum) by
          funext omega
          exact source.outputInContext_toSum omega]
    _ = abstainEvent source.actualOutput :=
      (abstainEvent_eq_erased source.actualOutput).symm

/-- The pre-cast exact-event proof transports to the recovery-context output by event equality. -/
theorem measurableSet_outputInContext_exactSelected
    (source : B2V3CanonicalSelectorSourceAdapter ctx) :
    MeasurableSet (ctx.exactSelected source.outputInContext) := by
  rw [source.outputInContext_exactSelected_eq]
  exact source.measurableSet_actualExactSelected

/-- The pre-cast wrong-event proof transports to the recovery-context output by event equality. -/
theorem measurableSet_outputInContext_wrongSelected
    (source : B2V3CanonicalSelectorSourceAdapter ctx)
    (hOutput : Measurable (fun omega => (source.actualOutput omega).toSum)) :
    MeasurableSet (wrongSelectedEvent source.outputInContext ctx.truth) := by
  rw [source.outputInContext_wrongSelected_eq]
  exact source.measurableSet_actualWrongSelected hOutput

/-- The pre-cast abstention proof transports to the recovery-context output by event equality. -/
theorem measurableSet_outputInContext_abstain
    (source : B2V3CanonicalSelectorSourceAdapter ctx)
    (hOutput : Measurable (fun omega => (source.actualOutput omega).toSum)) :
    MeasurableSet (abstainEvent source.outputInContext) := by
  rw [source.outputInContext_abstain_eq]
  exact measurableSet_abstainEvent source.actualOutput hOutput

end B2V3CanonicalSelectorSourceAdapter

/--
Canonical conformance is stronger than erased-output equality alone: the named success premise is
what licenses the universal selector-success event and zero selector-failure budget.
-/
structure B2V3CanonicalConformance
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    [MeasurableSpace (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}
    (source : B2V3CanonicalSelectorSourceAdapter ctx)
    (etaTie etaGuard etaNum : ℝ) : Prop where
  erasedOutput_eq : ∀ omega,
    (source.actualOutput omega).toSum =
      source.canonicalReferenceErasedOutput etaTie etaGuard etaNum omega
  selectorSuccess_eq_univ :
    ctx.selectorSuccess source.outputInContext etaTie etaGuard etaNum = Set.univ

/-- Pointwise erased-output conformance transports canonical reference measurability. -/
theorem measurable_actualErasedOutput_of_canonicalConformance
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    [MeasurableSpace (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}
    (source : B2V3CanonicalSelectorSourceAdapter ctx)
    (etaTie etaGuard etaNum : ℝ)
    (hConformance : B2V3CanonicalConformance source etaTie etaGuard etaNum) :
    Measurable (fun omega => (source.actualOutput omega).toSum) := by
  rw [show (fun omega => (source.actualOutput omega).toSum) =
    source.canonicalReferenceErasedOutput etaTie etaGuard etaNum by
      funext omega
      exact hConformance.erasedOutput_eq omega]
  exact source.measurable_canonicalReferenceErasedOutput etaTie etaGuard etaNum

/-- Canonical conformance also makes the pool-transported context output measurable. -/
theorem measurable_outputInContext_toSum_of_canonicalConformance
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    [MeasurableSpace (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}
    (source : B2V3CanonicalSelectorSourceAdapter ctx)
    (etaTie etaGuard etaNum : ℝ)
    (hConformance : B2V3CanonicalConformance source etaTie etaGuard etaNum) :
    Measurable (fun omega => (source.outputInContext omega).toSum) := by
  exact source.measurable_outputInContext_toSum
    (measurable_actualErasedOutput_of_canonicalConformance source etaTie etaGuard etaNum
      hConformance)

/-- Strong canonical conformance yields exactly the universal success event. -/
theorem selectorSuccess_eq_univ_of_canonicalConformance
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    [MeasurableSpace (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}
    (source : B2V3CanonicalSelectorSourceAdapter ctx)
    (etaTie etaGuard etaNum : ℝ)
    (hConformance : B2V3CanonicalConformance source etaTie etaGuard etaNum) :
    ctx.selectorSuccess source.outputInContext etaTie etaGuard etaNum = Set.univ :=
  hConformance.selectorSuccess_eq_univ

/-- The universal-success conformance path has zero selector-failure mass. -/
theorem measure_selectorFailure_eq_zero_of_canonicalConformance
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    [MeasurableSpace (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}
    (source : B2V3CanonicalSelectorSourceAdapter ctx)
    (etaTie etaGuard etaNum : ℝ)
    (hConformance : B2V3CanonicalConformance source etaTie etaGuard etaNum) :
    ctx.jointLaw (ctx.selectorSuccess source.outputInContext etaTie etaGuard etaNum)ᶜ = 0 := by
  rw [hConformance.selectorSuccess_eq_univ]
  simp

/-- Fallible or numerical source-output path retaining its named measurable success event. -/
structure B2V3FallibleSelectorOutput
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    [MeasurableSpace (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (output : ctx.SelectorOutput) (etaTie etaGuard etaNum : ℝ) (deltaO : ENNReal) where
  O_actual : Set Omega
  measurable_O_actual : MeasurableSet O_actual
  subset_selectorSuccess : O_actual ⊆ ctx.selectorSuccess output etaTie etaGuard etaNum
  selectorFailure_le : ctx.jointLaw O_actualᶜ ≤ deltaO

end BiostatFormal.HighDimensional.Multiview
