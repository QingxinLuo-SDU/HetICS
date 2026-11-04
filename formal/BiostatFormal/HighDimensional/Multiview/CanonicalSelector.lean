import BiostatFormal.HighDimensional.Multiview.CanonicalPartitionKey
import BiostatFormal.HighDimensional.Multiview.SelectorContract
import Mathlib.MeasureTheory.Measure.MeasureSpaceDef

/-!
# Canonical deterministic reference selector

This module proves the list-choice properties of `canonicalFinalChoice`, then wraps that existing
choice rule as a total deterministic selector.  Nonempty ordered inputs select an empirically
near-optimal payload under nonnegative tolerances; empty inputs abstain only with `emptyPool`.
-/

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

universe uOmega uLogicalId uCandidate uEvaluationContext
universe uCenter uLabel uSlot uRecordContext

/-- A value returned by `choosePreferred` came from its input list. -/
theorem choosePreferred_mem
    {α : Type*} (before : α → α → Bool)
    {values : List α} {chosen : α} :
    choosePreferred before values = some chosen → chosen ∈ values := by
  induction values with
  | nil => simp [choosePreferred]
  | cons value rest ih =>
      cases hRest : choosePreferred before rest with
      | none =>
          intro hChosen
          simp [choosePreferred, hRest] at hChosen
          subst chosen
          simp
      | some best =>
          by_cases hBefore : before best value
          · intro hChosen
            simp [choosePreferred, hRest, hBefore] at hChosen
            subst chosen
            exact List.mem_cons_of_mem value (ih hRest)
          · intro hChosen
            simp [choosePreferred, hRest, hBefore] at hChosen
            subst chosen
            exact List.mem_cons_self

/-- A nonempty list always has a realized minimum risk. -/
theorem minimumCandidateRisk?_eq_some_of_nonempty
    {E : ℕ} {LogicalId : Type uLogicalId} {Payload : Type uCandidate}
    (records : List (CanonicalCandidateRecord E LogicalId Payload)) :
    records ≠ [] → ∃ minimumRisk,
      minimumCandidateRisk? records = some minimumRisk := by
  cases records with
  | nil => simp
  | cons record rest =>
      intro _
      cases hMinimum : minimumCandidateRisk? rest with
      | none =>
          exact ⟨record.risk, by simp [minimumCandidateRisk?, hMinimum]⟩
      | some minimumRisk =>
          exact ⟨min record.risk minimumRisk,
            by simp [minimumCandidateRisk?, hMinimum]⟩

/-- The realized list minimum is no larger than every member risk. -/
theorem minimumCandidateRisk?_le_of_mem
    {E : ℕ} {LogicalId : Type uLogicalId} {Payload : Type uCandidate}
    {records : List (CanonicalCandidateRecord E LogicalId Payload)}
    {minimumRisk : ℝ}
    {record : CanonicalCandidateRecord E LogicalId Payload} :
    minimumCandidateRisk? records = some minimumRisk →
    record ∈ records → minimumRisk ≤ record.risk := by
  induction records generalizing minimumRisk record with
  | nil => simp
  | cons head tail ih =>
      intro hMinimum hRecord
      cases hTail : minimumCandidateRisk? tail with
      | none =>
          have hEq : head.risk = minimumRisk := by
            rw [minimumCandidateRisk?, hTail] at hMinimum
            exact Option.some.inj hMinimum
          rcases List.mem_cons.mp hRecord with rfl | hRecord
          · exact le_of_eq hEq.symm
          · have hTailNonempty : tail ≠ [] := by
              intro hEmpty
              subst tail
              exact List.not_mem_nil hRecord
            obtain ⟨tailMinimum, hTailMinimum⟩ :=
              minimumCandidateRisk?_eq_some_of_nonempty tail hTailNonempty
            rw [hTail] at hTailMinimum
            cases hTailMinimum
      | some tailMinimum =>
          have hEq : min head.risk tailMinimum = minimumRisk := by
            rw [minimumCandidateRisk?, hTail] at hMinimum
            exact Option.some.inj hMinimum
          rw [← hEq]
          rcases List.mem_cons.mp hRecord with rfl | hRecord
          · exact min_le_left _ _
          · exact (min_le_right _ _).trans (ih hTail hRecord)

/-- The realized list minimum is the risk of an actual input record. -/
theorem exists_mem_risk_eq_minimumCandidateRisk
    {E : ℕ} {LogicalId : Type uLogicalId} {Payload : Type uCandidate}
    {records : List (CanonicalCandidateRecord E LogicalId Payload)}
    {minimumRisk : ℝ} :
    minimumCandidateRisk? records = some minimumRisk →
    ∃ record ∈ records, record.risk = minimumRisk := by
  induction records generalizing minimumRisk with
  | nil => simp [minimumCandidateRisk?]
  | cons head tail ih =>
      intro hMinimum
      cases hTail : minimumCandidateRisk? tail with
      | none =>
          have hEq : head.risk = minimumRisk := by
            rw [minimumCandidateRisk?, hTail] at hMinimum
            exact Option.some.inj hMinimum
          exact ⟨head, List.mem_cons_self, hEq⟩
      | some tailMinimum =>
          have hEq : min head.risk tailMinimum = minimumRisk := by
            rw [minimumCandidateRisk?, hTail] at hMinimum
            exact Option.some.inj hMinimum
          rcases le_total head.risk tailMinimum with hHead | hTailLe
          · exact ⟨head, List.mem_cons_self,
              (min_eq_left hHead).symm.trans hEq⟩
          · obtain ⟨record, hRecord, hRisk⟩ := ih hTail
            exact ⟨record, List.mem_cons_of_mem head hRecord,
              hRisk.trans ((min_eq_right hTailLe).symm.trans hEq)⟩

/-- A value returned by the canonical final choice came from its ordered input. -/
theorem canonicalFinalChoice_mem
    {E : ℕ} {LogicalId : Type uLogicalId} {Payload : Type uCandidate}
    {records : List (CanonicalCandidateRecord E LogicalId Payload)}
    {tolerance : ℝ}
    {chosen : CanonicalCandidateRecord E LogicalId Payload} :
    canonicalFinalChoice records tolerance = some chosen → chosen ∈ records := by
  intro hChosen
  cases hMinimum : minimumCandidateRisk? records with
  | none => simp [canonicalFinalChoice, hMinimum] at hChosen
  | some minimumRisk =>
      have hFiltered : chosen ∈ records.filter
          (fun record => record.risk ≤ minimumRisk + tolerance) :=
        choosePreferred_mem canonicalRecordBefore (by
          simpa [canonicalFinalChoice, hMinimum] using hChosen)
      exact (List.mem_filter.mp hFiltered).1

private theorem choosePreferred_ne_none_of_nonempty
    {α : Type*} (before : α → α → Bool) {values : List α} :
    values ≠ [] → choosePreferred before values ≠ none := by
  cases values with
  | nil => simp
  | cons value rest =>
      intro _
      cases hRest : choosePreferred before rest with
      | none => simp [choosePreferred, hRest]
      | some best =>
          by_cases hBefore : before best value <;>
            simp [choosePreferred, hRest, hBefore]

/-- Nonnegative tolerance keeps at least one minimum-risk record eligible. -/
theorem canonicalFinalChoice_ne_none_of_nonempty
    {E : ℕ} {LogicalId : Type uLogicalId} {Payload : Type uCandidate}
    (records : List (CanonicalCandidateRecord E LogicalId Payload))
    (tolerance : ℝ) (hTolerance : 0 ≤ tolerance) :
    records ≠ [] → canonicalFinalChoice records tolerance ≠ none := by
  intro hRecords
  obtain ⟨minimumRisk, hMinimum⟩ :=
    minimumCandidateRisk?_eq_some_of_nonempty records hRecords
  obtain ⟨record, hRecord, hRisk⟩ :=
    exists_mem_risk_eq_minimumCandidateRisk hMinimum
  have hFiltered : record ∈ records.filter
      (fun candidate => candidate.risk ≤ minimumRisk + tolerance) := by
    refine List.mem_filter.mpr ⟨hRecord, ?_⟩
    apply decide_eq_true
    linarith
  have hFilteredNonempty :
      records.filter (fun candidate => candidate.risk ≤ minimumRisk + tolerance) ≠ [] := by
    intro hEmpty
    rw [hEmpty] at hFiltered
    exact List.not_mem_nil hFiltered
  simpa [canonicalFinalChoice, hMinimum] using
    (choosePreferred_ne_none_of_nonempty canonicalRecordBefore hFilteredNonempty)

/-- Every canonical choice is within tolerance of every realized input risk. -/
theorem canonicalFinalChoice_risk_le
    {E : ℕ} {LogicalId : Type uLogicalId} {Payload : Type uCandidate}
    {records : List (CanonicalCandidateRecord E LogicalId Payload)}
    {tolerance : ℝ} (hTolerance : 0 ≤ tolerance)
    {chosen : CanonicalCandidateRecord E LogicalId Payload} :
    canonicalFinalChoice records tolerance = some chosen →
    ∀ record ∈ records, chosen.risk ≤ record.risk + tolerance := by
  intro hChosen
  cases hMinimum : minimumCandidateRisk? records with
  | none => simp [canonicalFinalChoice, hMinimum] at hChosen
  | some minimumRisk =>
      have hFiltered : chosen ∈ records.filter
          (fun record => record.risk ≤ minimumRisk + tolerance) :=
        choosePreferred_mem canonicalRecordBefore (by
          simpa [canonicalFinalChoice, hMinimum] using hChosen)
      have hChosenRisk : chosen.risk ≤ minimumRisk + tolerance :=
        of_decide_eq_true (List.mem_filter.mp hFiltered).2
      intro record hRecord
      have hMinimumRisk : minimumRisk ≤ record.risk :=
        minimumCandidateRisk?_le_of_mem hMinimum hRecord
      by_cases hNegative : tolerance < 0
      · exact (not_lt_of_ge hTolerance hNegative).elim
      · linarith

/--
Frozen ordered canonical records together with a payload-level pool and the shared empirical-risk
interpretation.  Payload uniqueness makes the list-to-finset projection an unambiguous guard.
-/
structure CanonicalSelectorInput
    (E : ℕ) (LogicalId : Type uLogicalId) (Candidate : Type uCandidate)
    (EvaluationContext : Type uEvaluationContext)
    [DecidableEq Candidate] [RiskInput Candidate]
    (spec : SelectorRiskSpec Candidate EvaluationContext) where
  records : List (CanonicalCandidateRecord E LogicalId Candidate)
  nodupPayload : (records.map CanonicalCandidateRecord.payload).Nodup
  empiricalRisk_eq :
    ∀ record, record ∈ records →
      record.risk = empiricalCompositeRisk spec.risk record.payload

namespace CanonicalSelectorInput

/-- The realized guarded pool is the finite payload projection of the frozen record list. -/
def pool
    {E : ℕ} {LogicalId : Type uLogicalId} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (input : CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec) :
    GuardedCandidatePool Candidate :=
  (input.records.map CanonicalCandidateRecord.payload).toFinset

/-- Every original canonical record contributes its payload to the guarded pool. -/
theorem payload_mem_pool
    {E : ℕ} {LogicalId : Type uLogicalId} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (input : CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec)
    {record : CanonicalCandidateRecord E LogicalId Candidate}
    (hRecord : record ∈ input.records) :
    record.payload ∈ input.pool := by
  simp only [pool, List.mem_toFinset, List.mem_map]
  exact ⟨record, hRecord, rfl⟩

/-- Every guarded payload has an original canonical record witness in the frozen list. -/
theorem exists_record_payload_eq_of_mem_pool
    {E : ℕ} {LogicalId : Type uLogicalId} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (input : CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec)
    {payload : Candidate} (hPayload : payload ∈ input.pool) :
    ∃ record ∈ input.records, record.payload = payload := by
  have hList : payload ∈ input.records.map CanonicalCandidateRecord.payload := by
    simpa only [pool, List.mem_toFinset] using hPayload
  rcases List.mem_map.mp hList with ⟨record, hRecord, hEq⟩
  exact ⟨record, hRecord, hEq⟩

end CanonicalSelectorInput

/--
The deterministic reference output.  It reuses `canonicalFinalChoice` exactly and has only the
`emptyPool` abstention branch.
-/
noncomputable def canonicalSelectorOutput
    {E : ℕ} {LogicalId : Type uLogicalId} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (input : CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec)
    (etaTie etaGuard etaNum : ℝ) :
    AlgorithmOutput input.pool SelectorAbstainReason :=
  match h : canonicalFinalChoice input.records
      (selectionTolerance etaTie etaGuard etaNum) with
  | none => .abstain .emptyPool
  | some record =>
      .selected ⟨record.payload,
        input.payload_mem_pool (canonicalFinalChoice_mem h)⟩

/-- Empty ordered input returns the sole reference-selector abstention reason. -/
theorem canonicalSelectorOutput_empty
    {E : ℕ} {LogicalId : Type uLogicalId} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (input : CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec)
    (etaTie etaGuard etaNum : ℝ) (hEmpty : input.records = []) :
    canonicalSelectorOutput input etaTie etaGuard etaNum = .abstain .emptyPool := by
  unfold canonicalSelectorOutput
  split
  · rfl
  · rename_i record hChoice
    rw [hEmpty, canonicalFinalChoice_empty] at hChoice
    cases hChoice

/-- A nonempty realized payload pool produces an actual selected subtype. -/
theorem canonicalSelectorOutput_selected_of_nonempty
    {E : ℕ} {LogicalId : Type uLogicalId} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (input : CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec)
    (etaTie etaGuard etaNum : ℝ)
    (hTie : 0 ≤ etaTie) (hGuard : 0 ≤ etaGuard) (hNum : 0 ≤ etaNum)
    (hPool : input.pool.Nonempty) :
    ∃ selected, canonicalSelectorOutput input etaTie etaGuard etaNum = .selected selected := by
  rcases hPool with ⟨payload, hPayload⟩
  rcases input.exists_record_payload_eq_of_mem_pool hPayload with
    ⟨record, hRecord, _⟩
  have hRecords : input.records ≠ [] := by
    intro hEmpty
    rw [hEmpty] at hRecord
    exact List.not_mem_nil hRecord
  have hTolerance : 0 ≤ selectionTolerance etaTie etaGuard etaNum :=
    selectionTolerance_nonneg hTie hGuard hNum
  have hNotNone := canonicalFinalChoice_ne_none_of_nonempty
    input.records (selectionTolerance etaTie etaGuard etaNum) hTolerance hRecords
  unfold canonicalSelectorOutput
  split
  · rename_i hChoice
    exact (hNotNone hChoice).elim
  · exact ⟨_, rfl⟩

/-- Selected output exposes the pool-membership proof already stored in its subtype. -/
theorem canonicalSelectorOutput_selected_mem
    {E : ℕ} {LogicalId : Type uLogicalId} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (input : CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec)
    (etaTie etaGuard etaNum : ℝ) {selected : GuardedPoolMember input.pool}
    (_hSelected : canonicalSelectorOutput input etaTie etaGuard etaNum = .selected selected) :
    selected.1 ∈ input.pool :=
  selected.2

/-- The deterministic reference selector never reports numerical failure. -/
theorem canonicalSelectorOutput_ne_numericalFailure
    {E : ℕ} {LogicalId : Type uLogicalId} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (input : CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec)
    (etaTie etaGuard etaNum : ℝ) :
    canonicalSelectorOutput input etaTie etaGuard etaNum ≠
      .abstain .numericalFailure := by
  unfold canonicalSelectorOutput
  split <;> simp

/-- Selected empirical risk is within the total tolerance of every guarded payload. -/
theorem canonicalSelectorOutput_nearOptimal
    {E : ℕ} {LogicalId : Type uLogicalId} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (input : CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec)
    (etaTie etaGuard etaNum : ℝ)
    (hTie : 0 ≤ etaTie) (hGuard : 0 ≤ etaGuard) (hNum : 0 ≤ etaNum)
    {selected : GuardedPoolMember input.pool}
    (hSelected : canonicalSelectorOutput input etaTie etaGuard etaNum = .selected selected) :
    ∀ candidate : GuardedPoolMember input.pool,
      empiricalCompositeRisk spec.risk selected.1 ≤
        empiricalCompositeRisk spec.risk candidate.1 +
          selectionTolerance etaTie etaGuard etaNum := by
  have hTolerance : 0 ≤ selectionTolerance etaTie etaGuard etaNum :=
    selectionTolerance_nonneg hTie hGuard hNum
  unfold canonicalSelectorOutput at hSelected
  split at hSelected
  · cases hSelected
  · rename_i chosen hChoice
    have hChosenMem : chosen ∈ input.records := canonicalFinalChoice_mem hChoice
    cases hSelected
    intro candidate
    rcases input.exists_record_payload_eq_of_mem_pool candidate.2 with
      ⟨record, hRecord, hPayload⟩
    have hRisk := canonicalFinalChoice_risk_le hTolerance hChoice record hRecord
    rw [input.empiricalRisk_eq chosen hChosenMem,
      input.empiricalRisk_eq record hRecord] at hRisk
    simpa [hPayload] using hRisk

/-- Zero total tolerance specializes near-optimality to exact empirical argmin behavior. -/
theorem canonicalSelectorOutput_exactArgmin_of_selectionTolerance_eq_zero
    {E : ℕ} {LogicalId : Type uLogicalId} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (input : CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec)
    (etaTie etaGuard etaNum : ℝ)
    (hTie : 0 ≤ etaTie) (hGuard : 0 ≤ etaGuard) (hNum : 0 ≤ etaNum)
    (hZero : selectionTolerance etaTie etaGuard etaNum = 0)
    {selected : GuardedPoolMember input.pool}
    (hSelected : canonicalSelectorOutput input etaTie etaGuard etaNum = .selected selected) :
    ∀ candidate : GuardedPoolMember input.pool,
      empiricalCompositeRisk spec.risk selected.1 ≤
        empiricalCompositeRisk spec.risk candidate.1 := by
  intro candidate
  have hNear := canonicalSelectorOutput_nearOptimal input etaTie etaGuard etaNum
    hTie hGuard hNum hSelected candidate
  simpa [hZero] using hNear

/-- Pointwise payload pools induced by a family of frozen ordered inputs. -/
def canonicalPool
    {Ω : Type uOmega} {E : ℕ} {LogicalId : Type uLogicalId}
    {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (input : Ω → CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec)
    (omega : Ω) : GuardedCandidatePool Candidate :=
  (input omega).pool

/-- Pointwise deterministic reference output induced by a random input family. -/
noncomputable def canonicalOutput
    {Ω : Type uOmega} {E : ℕ} {LogicalId : Type uLogicalId}
    {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (input : Ω → CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec)
    (etaTie etaGuard etaNum : ℝ) (omega : Ω) :
    AlgorithmOutput (canonicalPool input omega) SelectorAbstainReason :=
  canonicalSelectorOutput (input omega) etaTie etaGuard etaNum

/-- The canonical family satisfies the abstract selector contract under nonnegative tolerances. -/
theorem canonicalSelectorContract
    {Ω : Type uOmega} {E : ℕ} {LogicalId : Type uLogicalId}
    {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (input : Ω → CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec)
    (etaTie etaGuard etaNum : ℝ)
    (hTie : 0 ≤ etaTie) (hGuard : 0 ≤ etaGuard) (hNum : 0 ≤ etaNum) :
    SelectorContract (canonicalPool input) spec
      (canonicalOutput input etaTie etaGuard etaNum) etaTie etaGuard etaNum where
  selects_of_nonempty omega hPool := by
    exact canonicalSelectorOutput_selected_of_nonempty (input omega)
      etaTie etaGuard etaNum hTie hGuard hNum hPool
  nearOptimal_of_selected omega selected hSelected := by
    exact canonicalSelectorOutput_nearOptimal (input omega)
      etaTie etaGuard etaNum hTie hGuard hNum hSelected

/-- For contextual candidate payloads, the deterministic selector-success event is universal. -/
theorem canonicalSelectorSuccessEvent_eq_univ
    {Ω : Type uOmega} {E : ℕ} {LogicalId : Type uLogicalId}
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    {spec : SelectorRiskSpec
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext}
    (input : Ω → CanonicalSelectorInput E LogicalId
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext spec)
    (truth : Partition Center Label) (etaTie etaGuard etaNum : ℝ)
    (hTie : 0 ≤ etaTie) (hGuard : 0 ≤ etaGuard) (hNum : 0 ≤ etaNum) :
    selectorSuccessEvent (canonicalPool input) spec
      (canonicalOutput input etaTie etaGuard etaNum) truth
      etaTie etaGuard etaNum = Set.univ := by
  ext omega
  simp only [Set.mem_univ, iff_true, selectorSuccessEvent, Set.mem_union,
    Set.mem_compl_iff, selectedNearOptimalEvent, Set.mem_setOf_eq]
  by_cases hSurvival : omega ∈ exactSurvivalEvent (canonicalPool input) truth
  · right
    have hPool : (canonicalPool input omega).Nonempty := by
      rcases hSurvival with ⟨witness, _⟩
      exact ⟨witness.1, witness.2⟩
    let contract := canonicalSelectorContract input etaTie etaGuard etaNum
      hTie hGuard hNum
    obtain ⟨selected, hSelected⟩ :=
      contract.selects_of_nonempty omega hPool
    exact ⟨selected, hSelected,
      contract.nearOptimal_of_selected omega selected hSelected⟩
  · exact Or.inl hSurvival

/-- The complement of the deterministic selector-success event has zero measure. -/
theorem measure_canonicalSelectorFailure_eq_zero
    {Ω : Type uOmega} {E : ℕ} {LogicalId : Type uLogicalId}
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {RecordContext : Type uRecordContext}
    {EvaluationContext : Type uEvaluationContext}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    [MeasurableSpace Ω] (mu : Measure Ω)
    {spec : SelectorRiskSpec
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext}
    (input : Ω → CanonicalSelectorInput E LogicalId
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext spec)
    (truth : Partition Center Label) (etaTie etaGuard etaNum : ℝ)
    (hTie : 0 ≤ etaTie) (hGuard : 0 ≤ etaGuard) (hNum : 0 ≤ etaNum) :
    mu (selectorSuccessEvent (canonicalPool input) spec
      (canonicalOutput input etaTie etaGuard etaNum) truth
      etaTie etaGuard etaNum)ᶜ = 0 := by
  rw [canonicalSelectorSuccessEvent_eq_univ input truth etaTie etaGuard etaNum
    hTie hGuard hNum]
  simp

end BiostatFormal.HighDimensional.Multiview
