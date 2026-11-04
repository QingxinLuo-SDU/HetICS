import BiostatFormal.HighDimensional.Multiview.CanonicalSelector
import Mathlib.MeasureTheory.Constructions.BorelSpace.Real
import Mathlib.MeasureTheory.MeasurableSpace.Constructions
import Mathlib.MeasureTheory.MeasurableSpace.Instances

/-!
# Fixed finite-index selector measurability

This module erases the dependent guarded-pool membership proof from selector outputs and freezes
random canonical inputs behind one finite logical-index envelope.  The only observable tie data are
the canonical partition's first-occurrence list and the primary provenance's slot/direction ranks;
`canonicalRecordBefore_eq_tieCoordinateBefore` proves that this projection preserves the existing
production comparator exactly.

Pinned-library search (Mathlib revision `81a5d257c8e410db227a6665ed08f64fea08e997`):

* `measurable_inl`, `measurable_inr`, `measurableSet_sum_iff`,
  `measurableSet_range_inl`, and the Sum measurable-space construction are in
  `Mathlib.MeasureTheory.MeasurableSpace.Constructions`;
* `Measurable.forall`, `Measurable.exists`, `Measurable.ite`, and
  `MeasurableSet.iUnion` provide countable logical combinations and finite branching;
* `Measurable.min`, `measurableSet_le`, and `measurableSet_lt` provide the real-order layer;
* `Bool.instMeasurableSpace`, `Nat.instMeasurableSpace`, and their measurable-singleton instances
  are discrete in `Mathlib.MeasureTheory.MeasurableSpace.Instances`.

No matching selector-specific replacement was found in the pinned StatLib or SLT sources.  Pinned
Lean's `List.filter` accepts the Boolean predicate `eligible omega` directly, so no proposition-valued
API adaptation is needed.  `CanonicalTieCoordinate` is countable and receives the explicit discrete
measurable space; no measurable space is assigned to `CanonicalCandidateRecord`.

The output proof uses private finite states that mirror `minimumCandidateRisk?`, tolerance filtering,
and `choosePreferred`; `indexedOutputToSum_eq_canonicalOutput_toSum` proves pointwise equality with
the existing production selector before measurability is transferred.  Only the private
`Option Index` state and the finite abstention-reason branch are discrete.  Exact/wrong event lemmas
retain fixed-index selected-output and exactness pullbacks as named hypotheses, because equality of
two varying continuous candidate records is not inferred to be measurable.
-/

namespace BiostatFormal.HighDimensional.Multiview

universe uOmega uIndex uLogicalId uCandidate uEvaluationContext

/-- Global abstention reasons form a finite discrete probability-facing branch. -/
instance : MeasurableSpace SelectorAbstainReason := ⊤

instance : MeasurableSingletonClass SelectorAbstainReason := ⟨fun _ => trivial⟩

namespace AlgorithmOutput

/-- Erase the selected candidate's dependent pool-membership proof into a fixed Sum codomain. -/
def toSum
    {Candidate Reason : Type*} [DecidableEq Candidate]
    {pool : GuardedCandidatePool Candidate} :
    AlgorithmOutput pool Reason → Reason ⊕ Candidate
  | .selected candidate => Sum.inr candidate.1
  | .abstain reason => Sum.inl reason

@[simp]
theorem toSum_selected
    {Candidate Reason : Type*} [DecidableEq Candidate]
    {pool : GuardedCandidatePool Candidate}
    (candidate : GuardedPoolMember pool) :
    (AlgorithmOutput.selected candidate : AlgorithmOutput pool Reason).toSum =
      Sum.inr candidate.1 :=
  rfl

@[simp]
theorem toSum_abstain
    {Candidate Reason : Type*} [DecidableEq Candidate]
    {pool : GuardedCandidatePool Candidate} (reason : Reason) :
    (AlgorithmOutput.abstain reason : AlgorithmOutput pool Reason).toSum =
      Sum.inl reason :=
  rfl

end AlgorithmOutput

/-- The complete observable coordinate used by the existing canonical tie comparator. -/
structure CanonicalTieCoordinate where
  partitionFirstOccurrence : List Nat
  primaryProvenanceRank : Option (Nat × Nat)
  deriving DecidableEq

/-- The finite tie-coordinate product is countable. -/
instance : Countable CanonicalTieCoordinate :=
  Function.Injective.countable (f := fun coordinate : CanonicalTieCoordinate =>
      (coordinate.partitionFirstOccurrence, coordinate.primaryProvenanceRank)) (by
      intro left right h
      cases left
      cases right
      simp_all)

/-- The tie coordinate is countable and is therefore equipped with its discrete measurable space. -/
instance : MeasurableSpace CanonicalTieCoordinate := ⊤

/-- Singletons of the countable tie coordinate are measurable in its discrete space. -/
instance : MeasurableSingletonClass CanonicalTieCoordinate := ⟨fun _ => trivial⟩

/-- Project a canonical record to exactly the fields read by `canonicalRecordBefore`. -/
def CanonicalCandidateRecord.tieCoordinate
    {E : ℕ} {LogicalId Candidate : Type*}
    (record : CanonicalCandidateRecord E LogicalId Candidate) :
    CanonicalTieCoordinate :=
  { partitionFirstOccurrence := record.partitionKey.firstOccurrence
    primaryProvenanceRank :=
      (primaryProvenance record.provenance).map
        (fun provenance => (provenance.slotRank, provenance.directionRank)) }

/-- Canonical strict tie preference expressed only through the measurable tie coordinate. -/
def canonicalTieCoordinateBefore
    (left right : CanonicalTieCoordinate) : Bool :=
  if left.partitionFirstOccurrence < right.partitionFirstOccurrence then
    true
  else if left.partitionFirstOccurrence = right.partitionFirstOccurrence then
    match left.primaryProvenanceRank, right.primaryProvenanceRank with
    | some leftRank, some rightRank =>
        decide (leftRank.1 < rightRank.1 ∨
          (leftRank.1 = rightRank.1 ∧ leftRank.2 < rightRank.2))
    | some _, none => true
    | none, some _ => false
    | none, none => false
  else
    false

/-- The production record comparator factors exactly through `tieCoordinate`. -/
theorem canonicalRecordBefore_eq_tieCoordinateBefore
    {E : ℕ} {LogicalId Candidate : Type*}
    (left right : CanonicalCandidateRecord E LogicalId Candidate) :
    canonicalRecordBefore left right =
      canonicalTieCoordinateBefore left.tieCoordinate right.tieCoordinate := by
  by_cases hBefore :
      left.partitionKey.firstOccurrence < right.partitionKey.firstOccurrence
  · simp [canonicalRecordBefore, canonicalTieCoordinateBefore,
      CanonicalCandidateRecord.tieCoordinate, hBefore]
  by_cases hSame :
      left.partitionKey.firstOccurrence = right.partitionKey.firstOccurrence
  · cases hLeft : primaryProvenance left.provenance <;>
      cases hRight : primaryProvenance right.provenance <;>
      simp [canonicalRecordBefore, canonicalTieCoordinateBefore,
        CanonicalCandidateRecord.tieCoordinate, provenanceBefore,
        hSame, hLeft, hRight]
  · simp [canonicalRecordBefore, canonicalTieCoordinateBefore,
      CanonicalCandidateRecord.tieCoordinate, hBefore, hSame]

/--
A fixed finite logical-index presentation of every realized canonical selector input.  Eligibility
may vary with the realization, while the index type and frozen traversal order do not.
-/
structure FiniteCanonicalSelectorEnvelope
    (Omega : Type uOmega) (Index : Type uIndex) (E : Nat)
    (LogicalId : Type uLogicalId) (Candidate : Type uCandidate)
    (EvaluationContext : Type uEvaluationContext)
    [Fintype Index] [DecidableEq Index]
    [DecidableEq Candidate] [RiskInput Candidate]
    (spec : SelectorRiskSpec Candidate EvaluationContext) where
  orderedIndices : List Index
  nodupIndices : orderedIndices.Nodup
  complete : ∀ index, index ∈ orderedIndices
  eligible : Omega → Index → Bool
  record : Omega → Index → CanonicalCandidateRecord E LogicalId Candidate
  nodupEligiblePayload : ∀ omega,
    (((orderedIndices.filter (eligible omega)).map
      (fun index => (record omega index).payload))).Nodup
  empiricalRisk_eq : ∀ omega index,
    eligible omega index = true →
      (record omega index).risk =
        empiricalCompositeRisk spec.risk (record omega index).payload

namespace FiniteCanonicalSelectorEnvelope

/-- Filter the frozen index order and map its realized records into the existing canonical input. -/
def toInput
    {Omega : Type uOmega} {Index : Type uIndex} {E : Nat}
    {LogicalId : Type uLogicalId} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [Fintype Index] [DecidableEq Index]
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (envelope : FiniteCanonicalSelectorEnvelope Omega Index E
      LogicalId Candidate EvaluationContext spec)
    (omega : Omega) :
    CanonicalSelectorInput E LogicalId Candidate EvaluationContext spec where
  records := (envelope.orderedIndices.filter (envelope.eligible omega)).map
    (envelope.record omega)
  nodupPayload := by
    simpa [List.map_map, Function.comp_def] using
      envelope.nodupEligiblePayload omega
  empiricalRisk_eq := by
    intro record hRecord
    rcases List.mem_map.mp hRecord with ⟨index, hIndex, rfl⟩
    exact envelope.empiricalRisk_eq omega index (List.mem_filter.mp hIndex).2

end FiniteCanonicalSelectorEnvelope

/-- Explicit measurable coordinates sufficient for the finite canonical selector proof layer. -/
structure CanonicalSelectorMeasurabilityAssumptions
    {Omega : Type uOmega} {Index : Type uIndex} {E : Nat}
    {LogicalId : Type uLogicalId} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [Fintype Index] [DecidableEq Index]
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    [MeasurableSpace Omega] [MeasurableSpace Candidate]
    (envelope : FiniteCanonicalSelectorEnvelope Omega Index E
      LogicalId Candidate EvaluationContext spec) : Prop where
  eligible_measurable : ∀ index,
    Measurable (fun omega => envelope.eligible omega index)
  risk_measurable : ∀ index,
    Measurable (fun omega => (envelope.record omega index).risk)
  tieCoordinate_measurable : ∀ index,
    Measurable (fun omega => (envelope.record omega index).tieCoordinate)
  payload_measurable : ∀ index,
    Measurable (fun omega => (envelope.record omega index).payload)

/-! ## Private finite-state factorization of the production selector -/

/--
The running minimum carries a Boolean nonemptiness flag and an arbitrary zero value in the empty
state.  This fixed product is convenient for measurability and is converted back to the production
`Option` only in the factorization proof.
-/
private def indexedMinimumState {Index : Type*}
    (indices : List Index) (eligible : Index → Bool) (risk : Index → ℝ) : Bool × ℝ :=
  match indices with
  | [] => (false, 0)
  | index :: rest =>
      let tail := indexedMinimumState rest eligible risk
      if eligible index then
        if tail.1 then (true, min (risk index) tail.2) else (true, risk index)
      else
        tail

/-- Interpret the private running-minimum state as the production optional minimum. -/
private def indexedMinimumStateToOption (state : Bool × ℝ) : Option ℝ :=
  if state.1 then some state.2 else none

/-- The private minimum state has its flag set exactly when some frozen index is eligible. -/
private theorem indexedMinimumState_fst_eq_true_iff {Index : Type*}
    (indices : List Index) (eligible : Index → Bool) (risk : Index → ℝ) :
    (indexedMinimumState indices eligible risk).1 = true ↔
      ∃ index ∈ indices, eligible index = true := by
  induction indices with
  | nil => simp [indexedMinimumState]
  | cons index rest ih =>
      by_cases hEligible : eligible index = true
      · simp only [indexedMinimumState, hEligible, if_pos]
        split <;> simp [hEligible]
      · simp [indexedMinimumState, hEligible, ih]

/-- The private minimum is exactly `minimumCandidateRisk?` on the filtered production records. -/
private theorem indexedMinimumState_toOption_eq_minimumCandidateRisk
    {Index : Type*} {E : ℕ} {LogicalId Candidate : Type*}
    (indices : List Index) (eligible : Index → Bool)
    (record : Index → CanonicalCandidateRecord E LogicalId Candidate) :
    indexedMinimumStateToOption
        (indexedMinimumState indices eligible (fun index => (record index).risk)) =
      minimumCandidateRisk? ((indices.filter eligible).map record) := by
  induction indices with
  | nil => simp [indexedMinimumState, indexedMinimumStateToOption, minimumCandidateRisk?]
  | cons index rest ih =>
      by_cases hEligible : eligible index = true
      · simp only [indexedMinimumState, hEligible, if_pos, List.filter_cons_of_pos,
          List.map_cons, minimumCandidateRisk?]
        rw [← ih]
        by_cases hTail :
            (indexedMinimumState rest eligible
              (fun index => (record index).risk)).1 = true
        · simp [indexedMinimumStateToOption, hTail] at ih ⊢
        · simp [indexedMinimumStateToOption, hTail] at ih ⊢
      · have hEligibleFalse : eligible index = false := Bool.eq_false_of_not_eq_true hEligible
        simp [indexedMinimumState, hEligibleFalse, ih]

/--
Choose among qualifying frozen indices with the production selector's right-fold tie semantics.
The return type is fixed because membership proofs and realized record types are not retained.
-/
private noncomputable def indexedPreferred?
    {Index : Type*} (indices : List Index)
    (eligible : Index → Bool) (risk : Index → ℝ)
    (tieCoordinate : Index → CanonicalTieCoordinate) (threshold : ℝ) : Option Index :=
  match indices with
  | [] => none
  | index :: rest =>
      let tail := indexedPreferred? rest eligible risk tieCoordinate threshold
      if eligible index && decide (risk index ≤ threshold) then
        match tail with
        | none => some index
        | some best =>
            if canonicalTieCoordinateBefore (tieCoordinate best) (tieCoordinate index) then
              some best
            else
              some index
      else
        tail

/-- The fixed-index preferred choice maps exactly to `choosePreferred` on production records. -/
private theorem indexedPreferred_map_eq_choosePreferred
    {Index : Type*} {E : ℕ} {LogicalId Candidate : Type*}
    (indices : List Index) (eligible : Index → Bool)
    (record : Index → CanonicalCandidateRecord E LogicalId Candidate)
    (threshold : ℝ) :
    Option.map record
        (indexedPreferred? indices eligible
          (fun index => (record index).risk)
          (fun index => (record index).tieCoordinate) threshold) =
      choosePreferred canonicalRecordBefore
        (((indices.filter eligible).map record).filter
          (fun candidate => candidate.risk ≤ threshold)) := by
  induction indices with
  | nil => simp [indexedPreferred?, choosePreferred]
  | cons index rest ih =>
      by_cases hEligible : eligible index = true
      · by_cases hRisk : (record index).risk ≤ threshold
        · simp only [indexedPreferred?, hEligible, Bool.true_and, decide_eq_true_eq,
            hRisk, if_pos, List.filter_cons_of_pos, List.map_cons]
          simp only [choosePreferred]
          rw [← ih]
          cases hTail : indexedPreferred? rest eligible
              (fun index => (record index).risk)
              (fun index => (record index).tieCoordinate) threshold with
          | none => rfl
          | some best =>
              simp only [Option.map_some, canonicalRecordBefore_eq_tieCoordinateBefore]
              split <;> rfl
        · simp [indexedPreferred?, hEligible, hRisk, ih]
      · have hEligibleFalse : eligible index = false := Bool.eq_false_of_not_eq_true hEligible
        simp [indexedPreferred?, hEligibleFalse, ih]

/-- Fixed-index version of `canonicalFinalChoice`, used only as a proof factorization. -/
private noncomputable def indexedFinalChoice?
    {Index : Type*} (indices : List Index)
    (eligible : Index → Bool) (risk : Index → ℝ)
    (tieCoordinate : Index → CanonicalTieCoordinate) (tolerance : ℝ) : Option Index :=
  let minimum := indexedMinimumState indices eligible risk
  if minimum.1 then
    indexedPreferred? indices eligible risk tieCoordinate (minimum.2 + tolerance)
  else
    none

/-- The private fixed-index final choice maps exactly to the existing production final choice. -/
private theorem indexedFinalChoice_map_eq_canonicalFinalChoice
    {Index : Type*} {E : ℕ} {LogicalId Candidate : Type*}
    (indices : List Index) (eligible : Index → Bool)
    (record : Index → CanonicalCandidateRecord E LogicalId Candidate)
    (tolerance : ℝ) :
    Option.map record
        (indexedFinalChoice? indices eligible
          (fun index => (record index).risk)
          (fun index => (record index).tieCoordinate) tolerance) =
      canonicalFinalChoice ((indices.filter eligible).map record) tolerance := by
  rw [canonicalFinalChoice,
    ← indexedMinimumState_toOption_eq_minimumCandidateRisk indices eligible record]
  unfold indexedFinalChoice? indexedMinimumStateToOption
  by_cases hMinimum :
      (indexedMinimumState indices eligible (fun index => (record index).risk)).1 = true
  · simp only [hMinimum, if_pos, Option.map]
    exact indexedPreferred_map_eq_choosePreferred indices eligible record _
  · simp [hMinimum]

/-- The Boolean comparator on the countable discrete tie coordinates is measurable. -/
private theorem measurable_canonicalTieCoordinateBefore :
    Measurable (fun coordinates : CanonicalTieCoordinate × CanonicalTieCoordinate =>
      canonicalTieCoordinateBefore coordinates.1 coordinates.2) :=
  measurable_of_countable _

/-- The finite running-minimum state is measurable from fixed-index eligibility and risk maps. -/
private theorem measurable_indexedMinimumState
    {Omega Index : Type*} [MeasurableSpace Omega]
    (indices : List Index) (eligible : Omega → Index → Bool)
    (risk : Omega → Index → ℝ)
    (hEligible : ∀ index, Measurable (fun omega => eligible omega index))
    (hRisk : ∀ index, Measurable (fun omega => risk omega index)) :
    Measurable (fun omega => indexedMinimumState indices (eligible omega) (risk omega)) := by
  induction indices with
  | nil => exact measurable_const
  | cons index rest ih =>
      have hEligibleSet : MeasurableSet {omega | eligible omega index = true} :=
        (hEligible index) (measurableSet_singleton true)
      have hTailSet : MeasurableSet
          {omega | (indexedMinimumState rest (eligible omega) (risk omega)).1 = true} :=
        ih.fst (measurableSet_singleton true)
      have hNonempty : Measurable (fun omega =>
          if (indexedMinimumState rest (eligible omega) (risk omega)).1 then
            (true, min (risk omega index)
              (indexedMinimumState rest (eligible omega) (risk omega)).2)
          else
            (true, risk omega index)) := by
        exact Measurable.ite hTailSet
          (measurable_const.prodMk ((hRisk index).min ih.snd))
          (measurable_const.prodMk (hRisk index))
      simpa only [indexedMinimumState] using
        Measurable.ite hEligibleSet hNonempty ih

/-- The finite preferred-index state is measurable from the source coordinate maps. -/
private theorem measurable_indexedPreferred?
    {Omega Index : Type*} [MeasurableSpace Omega]
    [Fintype Index] [DecidableEq Index]
    [MeasurableSpace (Option Index)]
    [MeasurableSingletonClass (Option Index)]
    (indices : List Index) (eligible : Omega → Index → Bool)
    (risk : Omega → Index → ℝ)
    (tieCoordinate : Omega → Index → CanonicalTieCoordinate)
    (threshold : Omega → ℝ)
    (hEligible : ∀ index, Measurable (fun omega => eligible omega index))
    (hRisk : ∀ index, Measurable (fun omega => risk omega index))
    (hTie : ∀ index, Measurable (fun omega => tieCoordinate omega index))
    (hThreshold : Measurable threshold) :
    Measurable (fun omega => indexedPreferred? indices (eligible omega) (risk omega)
      (tieCoordinate omega) (threshold omega)) := by
  induction indices with
  | nil => exact measurable_const
  | cons index rest ih =>
      have hEligibleSet : MeasurableSet {omega | eligible omega index = true} :=
        (hEligible index) (measurableSet_singleton true)
      have hRiskSet : MeasurableSet {omega | risk omega index ≤ threshold omega} :=
        measurableSet_le (hRisk index) hThreshold
      have hQualified : MeasurableSet
          {omega | (eligible omega index &&
            decide (risk omega index ≤ threshold omega)) = true} := by
        have hSetEq :
            {omega | (eligible omega index &&
              decide (risk omega index ≤ threshold omega)) = true} =
              {omega | eligible omega index = true} ∩
                {omega | risk omega index ≤ threshold omega} := by
          ext omega
          simp
        rw [hSetEq]
        exact hEligibleSet.inter hRiskSet
      have hStep : Measurable (fun pair : Omega × Option Index =>
          match pair.2 with
          | none => some index
          | some best =>
              if canonicalTieCoordinateBefore
                  (tieCoordinate pair.1 best) (tieCoordinate pair.1 index) then
                some best
              else
                some index) := by
        apply measurable_from_prod_countable_left
        intro choice
        cases choice with
        | none => exact measurable_const
        | some best =>
            have hBefore : Measurable (fun omega => canonicalTieCoordinateBefore
                (tieCoordinate omega best) (tieCoordinate omega index)) :=
              measurable_canonicalTieCoordinateBefore.comp
                ((hTie best).prodMk (hTie index))
            have hBeforeSet : MeasurableSet {omega | canonicalTieCoordinateBefore
                (tieCoordinate omega best) (tieCoordinate omega index) = true} :=
              hBefore (measurableSet_singleton true)
            exact Measurable.ite hBeforeSet measurable_const measurable_const
      have hChosen : Measurable (fun omega =>
          match indexedPreferred? rest (eligible omega) (risk omega)
              (tieCoordinate omega) (threshold omega) with
          | none => some index
          | some best =>
              if canonicalTieCoordinateBefore
                  (tieCoordinate omega best) (tieCoordinate omega index) then
                some best
              else
                some index) :=
        hStep.comp (measurable_id.prodMk ih)
      simpa only [indexedPreferred?] using
        Measurable.ite hQualified hChosen ih

/-- The exact fixed-index factor of `canonicalFinalChoice` is measurable. -/
private theorem measurable_indexedFinalChoice?
    {Omega Index : Type*} [MeasurableSpace Omega]
    [Fintype Index] [DecidableEq Index]
    [MeasurableSpace (Option Index)]
    [MeasurableSingletonClass (Option Index)]
    (indices : List Index) (eligible : Omega → Index → Bool)
    (risk : Omega → Index → ℝ)
    (tieCoordinate : Omega → Index → CanonicalTieCoordinate)
    (tolerance : ℝ)
    (hEligible : ∀ index, Measurable (fun omega => eligible omega index))
    (hRisk : ∀ index, Measurable (fun omega => risk omega index))
    (hTie : ∀ index, Measurable (fun omega => tieCoordinate omega index)) :
    Measurable (fun omega => indexedFinalChoice? indices (eligible omega) (risk omega)
      (tieCoordinate omega) tolerance) := by
  have hMinimum := measurable_indexedMinimumState indices eligible risk hEligible hRisk
  have hMinimumSet : MeasurableSet
      {omega | (indexedMinimumState indices (eligible omega) (risk omega)).1 = true} :=
    hMinimum.fst (measurableSet_singleton true)
  have hPreferred := measurable_indexedPreferred? indices eligible risk tieCoordinate
    (fun omega => (indexedMinimumState indices (eligible omega) (risk omega)).2 + tolerance)
    hEligible hRisk hTie (hMinimum.snd.add_const tolerance)
  simpa only [indexedFinalChoice?] using
    Measurable.ite hMinimumSet hPreferred measurable_const

/-- Erased output assembled from the fixed-index factor, with no dependent membership proof. -/
private noncomputable def indexedOutputToSum
    {Index Candidate : Type*} (indices : List Index)
    (eligible : Index → Bool) (risk : Index → ℝ)
    (tieCoordinate : Index → CanonicalTieCoordinate)
    (payload : Index → Candidate) (tolerance : ℝ) :
    SelectorAbstainReason ⊕ Candidate :=
  match indexedFinalChoice? indices eligible risk tieCoordinate tolerance with
  | none => Sum.inl .emptyPool
  | some index => Sum.inr (payload index)

/-- The private erased output is measurable once every fixed-index payload map is measurable. -/
private theorem measurable_indexedOutputToSum
    {Omega Index Candidate : Type*}
    [MeasurableSpace Omega] [MeasurableSpace Candidate]
    [Fintype Index] [DecidableEq Index]
    (indices : List Index) (eligible : Omega → Index → Bool)
    (risk : Omega → Index → ℝ)
    (tieCoordinate : Omega → Index → CanonicalTieCoordinate)
    (payload : Omega → Index → Candidate) (tolerance : ℝ)
    (hEligible : ∀ index, Measurable (fun omega => eligible omega index))
    (hRisk : ∀ index, Measurable (fun omega => risk omega index))
    (hTie : ∀ index, Measurable (fun omega => tieCoordinate omega index))
    (hPayload : ∀ index, Measurable (fun omega => payload omega index)) :
    Measurable (fun omega => indexedOutputToSum indices (eligible omega) (risk omega)
      (tieCoordinate omega) (payload omega) tolerance) := by
  letI : MeasurableSpace (Option Index) := ⊤
  letI : MeasurableSingletonClass (Option Index) := ⟨fun _ => trivial⟩
  have hChoice := measurable_indexedFinalChoice? indices eligible risk tieCoordinate tolerance
    hEligible hRisk hTie
  have hAssemble : Measurable (fun pair : Omega × Option Index =>
      match pair.2 with
      | none => (Sum.inl .emptyPool : SelectorAbstainReason ⊕ Candidate)
      | some index => Sum.inr (payload pair.1 index)) := by
    apply measurable_from_prod_countable_left
    intro choice
    cases choice with
    | none => exact measurable_const
    | some index => exact measurable_inr.comp (hPayload index)
  exact hAssemble.comp (measurable_id.prodMk hChoice)

/-- The fixed-index erased helper is extensionally the existing production canonical output. -/
private theorem indexedOutputToSum_eq_canonicalOutput_toSum
    {Omega Index : Type*} {E : ℕ}
    {LogicalId Candidate EvaluationContext : Type*}
    [Fintype Index] [DecidableEq Index]
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    (envelope : FiniteCanonicalSelectorEnvelope Omega Index E
      LogicalId Candidate EvaluationContext spec)
    (etaTie etaGuard etaNum : ℝ) (omega : Omega) :
    indexedOutputToSum envelope.orderedIndices (envelope.eligible omega)
        (fun index => (envelope.record omega index).risk)
        (fun index => (envelope.record omega index).tieCoordinate)
        (fun index => (envelope.record omega index).payload)
        (selectionTolerance etaTie etaGuard etaNum) =
      (canonicalOutput envelope.toInput etaTie etaGuard etaNum omega).toSum := by
  have hChoice := indexedFinalChoice_map_eq_canonicalFinalChoice
    envelope.orderedIndices (envelope.eligible omega) (envelope.record omega)
    (selectionTolerance etaTie etaGuard etaNum)
  cases hIndex : indexedFinalChoice? envelope.orderedIndices (envelope.eligible omega)
      (fun index => (envelope.record omega index).risk)
      (fun index => (envelope.record omega index).tieCoordinate)
      (selectionTolerance etaTie etaGuard etaNum) with
  | none =>
      have hFinal : canonicalFinalChoice (envelope.toInput omega).records
          (selectionTolerance etaTie etaGuard etaNum) = none := by
        simpa [FiniteCanonicalSelectorEnvelope.toInput, hIndex] using hChoice.symm
      rw [indexedOutputToSum, hIndex]
      simp only
      unfold canonicalOutput canonicalPool canonicalSelectorOutput
      split
      · rfl
      · rename_i record hSome
        rw [hFinal] at hSome
        cases hSome
  | some index =>
      have hFinal : canonicalFinalChoice (envelope.toInput omega).records
          (selectionTolerance etaTie etaGuard etaNum) = some (envelope.record omega index) := by
        simpa [FiniteCanonicalSelectorEnvelope.toInput, hIndex] using hChoice.symm
      rw [indexedOutputToSum, hIndex]
      simp only
      unfold canonicalOutput canonicalPool canonicalSelectorOutput
      split
      · rename_i hNone
        rw [hFinal] at hNone
        cases hNone
      · rename_i chosen hSome
        have hChosen : chosen = envelope.record omega index :=
          Option.some.inj (hSome.symm.trans hFinal)
        subst chosen
        rfl

/--
The erased output of the existing canonical selector is measurable from exactly the four explicit
source-coordinate families in `CanonicalSelectorMeasurabilityAssumptions`.
-/
theorem measurable_canonicalOutput_toSum
    {Omega : Type uOmega} {Index : Type uIndex} {E : ℕ}
    {LogicalId : Type uLogicalId} {Candidate : Type uCandidate}
    {EvaluationContext : Type uEvaluationContext}
    [Fintype Index] [DecidableEq Index]
    [DecidableEq Candidate] [RiskInput Candidate]
    {spec : SelectorRiskSpec Candidate EvaluationContext}
    [MeasurableSpace Omega] [MeasurableSpace Candidate]
    (envelope : FiniteCanonicalSelectorEnvelope Omega Index E
      LogicalId Candidate EvaluationContext spec)
    (etaTie etaGuard etaNum : ℝ)
    (hCoordinates : CanonicalSelectorMeasurabilityAssumptions envelope) :
    Measurable (fun omega =>
      (canonicalOutput envelope.toInput etaTie etaGuard etaNum omega).toSum) := by
  have hIndexed := measurable_indexedOutputToSum envelope.orderedIndices envelope.eligible
    (fun omega index => (envelope.record omega index).risk)
    (fun omega index => (envelope.record omega index).tieCoordinate)
    (fun omega index => (envelope.record omega index).payload)
    (selectionTolerance etaTie etaGuard etaNum)
    hCoordinates.eligible_measurable hCoordinates.risk_measurable
    hCoordinates.tieCoordinate_measurable hCoordinates.payload_measurable
  convert hIndexed using 1
  funext omega
  exact (indexedOutputToSum_eq_canonicalOutput_toSum
    envelope etaTie etaGuard etaNum omega).symm

/-! ## Selector-event measurability -/

/-- Abstention is the pullback of the finite discrete left branch of the erased output. -/
theorem measurableSet_abstainEvent
    {Omega : Type uOmega} {Candidate : Type uCandidate}
    [DecidableEq Candidate]
    [MeasurableSpace Omega] [MeasurableSpace Candidate]
    {pool : Omega → GuardedCandidatePool Candidate}
    (output : (omega : Omega) → AlgorithmOutput (pool omega) SelectorAbstainReason)
    (hOutput : Measurable (fun omega => (output omega).toSum)) :
    MeasurableSet (abstainEvent output) := by
  have hEventEq : abstainEvent output =
      (fun omega => (output omega).toSum) ⁻¹'
        Set.range (Sum.inl : SelectorAbstainReason →
          SelectorAbstainReason ⊕ Candidate) := by
    ext omega
    cases hState : output omega with
    | selected selected => simp [abstainEvent, hState]
    | abstain reason => simp [abstainEvent, hState]
  rw [hEventEq]
  exact hOutput measurableSet_range_inl

/--
Exact selection is a finite union of fixed-index selected-output events intersected with the
corresponding explicit exactness events.  Candidate equality measurability is not inferred.
-/
theorem measurableSet_exactSelectedEvent
    {Omega : Type uOmega} {Index : Type uIndex} {E : ℕ}
    {LogicalId : Type uLogicalId}
    {Center Label Slot RecordContext EvaluationContext : Type*}
    [Fintype Index] [DecidableEq Index]
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    [MeasurableSpace Omega]
    [MeasurableSpace (CandidateRecord Center Label Slot RecordContext)]
    {spec : SelectorRiskSpec
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext}
    (envelope : FiniteCanonicalSelectorEnvelope Omega Index E LogicalId
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext spec)
    (output : (omega : Omega) → AlgorithmOutput
      (envelope.toInput omega).pool SelectorAbstainReason)
    (truth : Partition Center Label)
    (hSelected : ∀ index, MeasurableSet {omega |
      (output omega).toSum = Sum.inr (envelope.record omega index).payload})
    (hExactness : ∀ index, MeasurableSet {omega |
      (envelope.record omega index).payload ∈ exactCandidateSet truth}) :
    MeasurableSet (exactSelectedEvent output truth) := by
  have hEventEq : exactSelectedEvent output truth =
      ⋃ index, ({omega |
        (output omega).toSum = Sum.inr (envelope.record omega index).payload} ∩
          {omega | (envelope.record omega index).payload ∈ exactCandidateSet truth}) := by
    ext omega
    constructor
    · rintro ⟨selected, hOutputState, hExact⟩
      obtain ⟨record, hRecord, hPayload⟩ :=
        (envelope.toInput omega).exists_record_payload_eq_of_mem_pool selected.2
      have hRecord' : record ∈
          (envelope.orderedIndices.filter (envelope.eligible omega)).map
            (envelope.record omega) := by
        simpa only [FiniteCanonicalSelectorEnvelope.toInput] using hRecord
      rcases List.mem_map.mp hRecord' with ⟨index, hIndex, hIndexRecord⟩
      have hIndexPayload : (envelope.record omega index).payload = selected.1 := by
        calc
          (envelope.record omega index).payload = record.payload :=
            congrArg CanonicalCandidateRecord.payload hIndexRecord
          _ = selected.1 := hPayload
      refine Set.mem_iUnion.mpr ⟨index, ?_⟩
      constructor
      · change (output omega).toSum = Sum.inr (envelope.record omega index).payload
        rw [hOutputState]
        simp [hIndexPayload]
      · simpa [hIndexPayload] using hExact
    · intro hUnion
      rcases Set.mem_iUnion.mp hUnion with ⟨index, hSelectedOutput, hExact⟩
      cases hOutputState : output omega with
      | selected selected =>
          have hPayload : selected.1 = (envelope.record omega index).payload := by
            simpa [hOutputState] using hSelectedOutput
          refine ⟨selected, hOutputState, ?_⟩
          simpa [hPayload] using hExact
      | abstain reason =>
          simp [hOutputState] at hSelectedOutput
  rw [hEventEq]
  exact MeasurableSet.iUnion fun index => (hSelected index).inter (hExactness index)

/--
Wrong selection is the measurable selected-output branch minus the measurable exact-selected
event.  This uses the same explicit fixed-index exactness boundary as the exact-event theorem.
-/
theorem measurableSet_wrongSelectedEvent
    {Omega : Type uOmega} {Index : Type uIndex} {E : ℕ}
    {LogicalId : Type uLogicalId}
    {Center Label Slot RecordContext EvaluationContext : Type*}
    [Fintype Index] [DecidableEq Index]
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    [MeasurableSpace Omega]
    [MeasurableSpace (CandidateRecord Center Label Slot RecordContext)]
    {spec : SelectorRiskSpec
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext}
    (envelope : FiniteCanonicalSelectorEnvelope Omega Index E LogicalId
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext spec)
    (output : (omega : Omega) → AlgorithmOutput
      (envelope.toInput omega).pool SelectorAbstainReason)
    (truth : Partition Center Label)
    (hOutput : Measurable (fun omega => (output omega).toSum))
    (hSelected : ∀ index, MeasurableSet {omega |
      (output omega).toSum = Sum.inr (envelope.record omega index).payload})
    (hExactness : ∀ index, MeasurableSet {omega |
      (envelope.record omega index).payload ∈ exactCandidateSet truth}) :
    MeasurableSet (wrongSelectedEvent output truth) := by
  have hExact := measurableSet_exactSelectedEvent envelope output truth hSelected hExactness
  have hSelectedBranch : MeasurableSet
      ((fun omega => (output omega).toSum) ⁻¹'
        Set.range (Sum.inr : CandidateRecord Center Label Slot RecordContext →
          SelectorAbstainReason ⊕ CandidateRecord Center Label Slot RecordContext)) :=
    hOutput measurableSet_range_inr
  have hEventEq : wrongSelectedEvent output truth =
      ((fun omega => (output omega).toSum) ⁻¹'
        Set.range (Sum.inr : CandidateRecord Center Label Slot RecordContext →
          SelectorAbstainReason ⊕ CandidateRecord Center Label Slot RecordContext)) \
        exactSelectedEvent output truth := by
    ext omega
    cases hState : output omega with
    | selected selected =>
        simp [wrongSelectedEvent, exactSelectedEvent, badCandidateSet,
          exactCandidateSet, hState]
    | abstain reason =>
        simp [wrongSelectedEvent, exactSelectedEvent, hState]
  rw [hEventEq]
  exact hSelectedBranch.diff hExact

/-- A selector-success event identified pointwise with `Set.univ` is measurable by rewriting. -/
theorem measurableSet_selectorSuccessEvent_of_eq_univ
    {Omega : Type uOmega} {Center Label Slot RecordContext EvaluationContext : Type*}
    [MeasurableSpace Omega]
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    (pool : Omega → GuardedCandidatePool
      (CandidateRecord Center Label Slot RecordContext))
    (spec : SelectorRiskSpec
      (CandidateRecord Center Label Slot RecordContext) EvaluationContext)
    (output : (omega : Omega) → AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label) (etaTie etaGuard etaNum : ℝ)
    (hEvent : selectorSuccessEvent pool spec output truth etaTie etaGuard etaNum = Set.univ) :
    MeasurableSet (selectorSuccessEvent pool spec output truth etaTie etaGuard etaNum) := by
  rw [hEvent]
  exact MeasurableSet.univ

end BiostatFormal.HighDimensional.Multiview
