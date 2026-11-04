import BiostatFormal.HighDimensional.Multiview.Guards
import Mathlib.Data.Finset.Sort

/-!
# Canonical partition and Ward tie keys

Raw numeric labels are not partition identities.  This file supplies a source-shaped
first-occurrence canonicalization, a fixed-center canonical partition record, provenance fields,
and an unordered Ward merge-pair key.  Every Ward step first normalizes pair orientation and then
minimizes the pair `(merge cost, unordered pair key)`.
-/

namespace BiostatFormal.HighDimensional.Multiview

/-- Source-shaped first-occurrence label canonicalization with an explicit `seen` list. -/
def canonicalizeAux {Label : Type*} [DecidableEq Label] : List Label → List Label → List ℕ
  | _, [] => []
  | seen, x :: xs =>
      if x ∈ seen then
        seen.idxOf x :: canonicalizeAux seen xs
      else
        seen.length :: canonicalizeAux (seen ++ [x]) xs

/-- Replace labels by their dense first-occurrence ranks `0,1,2,...`. -/
def canonicalizeLabels {Label : Type*} [DecidableEq Label] (labels : List Label) : List ℕ :=
  canonicalizeAux [] labels

/-- Minimum natural number, with an explicit zero sentinel for the empty set. -/
def minNatOrZero (values : Finset ℕ) : ℕ :=
  if h : values.Nonempty then values.min' h else 0

/-- Center indices in the same realized cluster as `i`. -/
def centerClassIndices
    {E : ℕ} {Label : Type*} [DecidableEq Label]
    (z : Partition (Fin E) Label) (i : Fin E) : Finset ℕ :=
  ((Finset.univ.filter fun j : Fin E => z j = z i).image Fin.val)

/-- The first center index carrying the same label as `i`. -/
def firstOccurrenceIndex
    {E : ℕ} {Label : Type*} [DecidableEq Label]
    (z : Partition (Fin E) Label) (i : Fin E) : ℕ :=
  minNatOrZero (centerClassIndices z i)

/--
Canonical partition identity.  `firstOccurrence` is the sortable source-facing key; the Boolean
same-cluster relation is retained so equality of keys exposes the estimand without interpreting
raw label names.
-/
structure CanonicalPartitionKey (E : ℕ) where
  firstOccurrence : List ℕ
  sameCluster : Fin E → Fin E → Bool

/-- Canonical key of a partition on the frozen center order `Fin E`. -/
def canonicalPartitionKey
    {E : ℕ} {Label : Type*} [DecidableEq Label]
    (z : Partition (Fin E) Label) : CanonicalPartitionKey E where
  firstOccurrence := List.ofFn (firstOccurrenceIndex z)
  sameCluster := fun i j => decide (z i = z j)

/-- A global label permutation cannot change the canonical partition identity. -/
theorem canonicalPartitionKey_relabel
    {E : ℕ} {Label : Type*} [DecidableEq Label]
    (σ : LabelPermutation Label) (z : Partition (Fin E) Label) :
    canonicalPartitionKey (relabel σ z) = canonicalPartitionKey z := by
  have hfirst : List.ofFn (firstOccurrenceIndex (relabel σ z)) =
      List.ofFn (firstOccurrenceIndex z)
      := by
    congr 1
    funext i
    apply congrArg minNatOrZero
    ext n
    simp [centerClassIndices, relabel]
  have hsame : (fun i j : Fin E => decide (relabel σ z i = relabel σ z j)) =
      fun i j => decide (z i = z j) := by
    funext i j
    simp [relabel]
  exact congrArg₂ (CanonicalPartitionKey.mk (E := E)) hfirst hsame

/-- Equal canonical keys imply exactly the same pairwise cluster relation. -/
theorem canonicalPartitionKey_eq_implies_sameCluster
    {E : ℕ} {Label₁ Label₂ : Type*} [DecidableEq Label₁] [DecidableEq Label₂]
    {z₁ : Partition (Fin E) Label₁} {z₂ : Partition (Fin E) Label₂}
    (h : canonicalPartitionKey z₁ = canonicalPartitionKey z₂) (i j : Fin E) :
    (z₁ i = z₁ j) ↔ (z₂ i = z₂ j) := by
  have hb := congrArg (fun key => key.sameCluster i j) h
  change decide (z₁ i = z₁ j) = decide (z₂ i = z₂ j) at hb
  constructor
  · intro hij
    have htrue : decide (z₂ i = z₂ j) = true := by
      rw [← hb]
      simp [hij]
    exact of_decide_eq_true htrue
  · intro hij
    have htrue : decide (z₁ i = z₁ j) = true := by
      rw [hb]
      simp [hij]
    exact of_decide_eq_true htrue

/-- Frozen logical provenance retained after partition-level deduplication. -/
structure PartitionProvenance (LogicalId : Type*) where
  slotRank : ℕ
  directionRank : ℕ
  logicalId : LogicalId
  deriving DecidableEq

/-- Final deterministic record key: partition identity first, provenance only after it agrees. -/
structure CanonicalCandidateKey (E : ℕ) (LogicalId : Type*) where
  partitionKey : CanonicalPartitionKey E
  provenance : PartitionProvenance LogicalId

/-- One risk-scored record before or after partition-level provenance aggregation. -/
structure CanonicalCandidateRecord (E : ℕ) (LogicalId Payload : Type*) where
  partitionKey : CanonicalPartitionKey E
  risk : ℝ
  provenance : List (PartitionProvenance LogicalId)
  payload : Payload

/--
Collect all provenance entries for one canonical partition in frozen input order.  This is the
deduplication rule: grouping changes neither the order nor the multiplicity of logical slot and
direction identifiers.
-/
def provenanceForPartitionKey
    {E : ℕ} {LogicalId Payload : Type*}
    (records : List (CanonicalCandidateRecord E LogicalId Payload))
    (key : CanonicalPartitionKey E) : List (PartitionProvenance LogicalId) :=
  (records.filter fun record =>
      record.partitionKey.firstOccurrence = key.firstOccurrence).flatMap
    CanonicalCandidateRecord.provenance

/-- Strict frozen provenance order: slot first, direction second. -/
def provenanceBefore {LogicalId : Type*}
    (left right : PartitionProvenance LogicalId) : Bool :=
  decide (left.slotRank < right.slotRank ∨
    (left.slotRank = right.slotRank ∧ left.directionRank < right.directionRank))

/-- Keep the earliest element when the supplied strict preference regards two values as tied. -/
def choosePreferred {α : Type*} (before : α → α → Bool) : List α → Option α
  | [] => none
  | value :: rest =>
      match choosePreferred before rest with
      | none => some value
      | some best => if before best value then some best else some value

/-- Primary frozen provenance representative of an aggregated partition group. -/
def primaryProvenance {LogicalId : Type*}
    (provenance : List (PartitionProvenance LogicalId)) : Option (PartitionProvenance LogicalId) :=
  choosePreferred provenanceBefore provenance

/-- Canonical final-record preference: partition key, then frozen slot/direction provenance. -/
def canonicalRecordBefore
    {E : ℕ} {LogicalId Payload : Type*}
    (left right : CanonicalCandidateRecord E LogicalId Payload) : Bool :=
  if left.partitionKey.firstOccurrence < right.partitionKey.firstOccurrence then
    true
  else if left.partitionKey.firstOccurrence = right.partitionKey.firstOccurrence then
    match primaryProvenance left.provenance, primaryProvenance right.provenance with
    | some leftProvenance, some rightProvenance => provenanceBefore leftProvenance rightProvenance
    | some _, none => true
    | none, some _ => false
    | none, none => false
  else
    false

/-- Minimum realized risk, with `none` rather than a fabricated value on an empty pool. -/
noncomputable def minimumCandidateRisk?
    {E : ℕ} {LogicalId Payload : Type*} :
    List (CanonicalCandidateRecord E LogicalId Payload) → Option ℝ
  | [] => none
  | record :: records =>
      match minimumCandidateRisk? records with
      | none => some record.risk
      | some minimumRisk => some (min record.risk minimumRisk)

/--
Final deterministic tie choice.  First form the risk-tolerance set, then use the canonical
partition key and only afterwards frozen slot/direction provenance.
-/
noncomputable def canonicalFinalChoice
    {E : ℕ} {LogicalId Payload : Type*}
    (records : List (CanonicalCandidateRecord E LogicalId Payload)) (riskTolerance : ℝ) :
    Option (CanonicalCandidateRecord E LogicalId Payload) :=
  match minimumCandidateRisk? records with
  | none => none
  | some minimumRisk =>
      choosePreferred canonicalRecordBefore
        (records.filter fun record => record.risk ≤ minimumRisk + riskTolerance)

@[simp]
theorem canonicalFinalChoice_empty
    {E : ℕ} {LogicalId Payload : Type*} (riskTolerance : ℝ) :
    canonicalFinalChoice
      ([] : List (CanonicalCandidateRecord E LogicalId Payload)) riskTolerance = none := by
  rfl

/-- Every output over an empty guarded pool is necessarily an explicit `ABSTAIN`. -/
theorem algorithmOutput_empty_eq_abstain
    {Candidate AbstainReason : Type*} [DecidableEq Candidate]
    (output : AlgorithmOutput (∅ : GuardedCandidatePool Candidate) AbstainReason) :
    ∃ reason, output = AlgorithmOutput.abstain reason := by
  cases output with
  | selected candidate =>
      exact (Finset.notMem_empty candidate.1 candidate.2).elim
  | abstain reason => exact ⟨reason, rfl⟩

/-- A Ward cluster is a finite set of centers in the frozen `Fin E` order. -/
abbrev WardCluster (E : ℕ) := Finset (Fin E)

/-- A value-level merge pair; validity remains a separate decidable predicate. -/
structure WardMergePair (E : ℕ) where
  left : WardCluster E
  right : WardCluster E
  deriving DecidableEq

/-- Canonical sorted center list of a Ward cluster. -/
def canonicalClusterKey {E : ℕ} (cluster : WardCluster E) : List ℕ :=
  (cluster.sort (· ≤ ·)).map Fin.val

/-- Unordered merge-pair key: the smaller cluster key is always first. -/
def canonicalMergePairKey {E : ℕ} (pair : WardMergePair E) : List (List ℕ) :=
  let leftKey := canonicalClusterKey pair.left
  let rightKey := canonicalClusterKey pair.right
  [min leftKey rightKey, max leftKey rightKey]

/-- Swap the two value-level sides of a Ward merge pair. -/
def WardMergePair.swap {E : ℕ} (pair : WardMergePair E) : WardMergePair E :=
  ⟨pair.right, pair.left⟩

@[simp]
theorem canonicalMergePairKey_swap {E : ℕ} (pair : WardMergePair E) :
    canonicalMergePairKey pair.swap = canonicalMergePairKey pair := by
  simp [canonicalMergePairKey, WardMergePair.swap, min_comm, max_comm]

/-- Normalize the orientation of a merge pair before it enters a tie set. -/
def normalizeWardMergePair {E : ℕ} (pair : WardMergePair E) : WardMergePair E :=
  if canonicalClusterKey pair.left ≤ canonicalClusterKey pair.right then pair else pair.swap

/-- Nonempty, disjoint, distinct-side validity for a value-level Ward pair. -/
def ValidWardMergePair {E : ℕ} (pair : WardMergePair E) : Prop :=
  pair.left.Nonempty ∧ pair.right.Nonempty ∧ Disjoint pair.left pair.right

/-- Enumerate each valid unordered cluster pair once using its canonical orientation. -/
noncomputable def wardMergePairs {E : ℕ}
    (pool : Finset (WardCluster E)) : Finset (WardMergePair E) := by
  classical
  exact ((pool.product pool).image fun pair => WardMergePair.mk pair.1 pair.2).filter fun pair =>
    ValidWardMergePair pair ∧ canonicalClusterKey pair.left < canonicalClusterKey pair.right

/-- Cost followed by the canonical unordered cluster-pair tie key. -/
def wardChoiceKey {E : ℕ} (mergeCost : WardMergePair E → ℝ) (pair : WardMergePair E) :
    ℝ × List (List ℕ) :=
  (mergeCost pair, canonicalMergePairKey pair)

/-- Binary center-set code used to order the already canonical unordered merge-pair keys. -/
def canonicalClusterCode {E : ℕ} (cluster : WardCluster E) : ℕ :=
  ∑ i ∈ cluster, 2 ^ i.val

/-- Cantor-style natural encoding of an unordered pair of binary cluster codes. -/
def canonicalMergePairRank {E : ℕ} (pair : WardMergePair E) : ℕ :=
  let leftCode := canonicalClusterCode pair.left
  let rightCode := canonicalClusterCode pair.right
  let lower := min leftCode rightCode
  let upper := max leftCode rightCode
  (lower + upper) * (lower + upper + 1) / 2 + upper

/--
Choose a Ward merge by minimizing `(cost, canonical unordered pair key)` after orientation
normalization.  Empty pair sets return `none` rather than fabricating a merge.
-/
noncomputable def canonicalWardStep
    {E : ℕ} (pairs : Finset (WardMergePair E))
    (mergeCost : WardMergePair E → ℝ) : Option (WardMergePair E) :=
  if hpairs : pairs.Nonempty then
    let normalized := pairs.image normalizeWardMergePair
    let hnormalized : normalized.Nonempty := Finset.image_nonempty.mpr hpairs
    let costs := normalized.image mergeCost
    let hcosts : costs.Nonempty := Finset.image_nonempty.mpr hnormalized
    let bestCost := costs.min' hcosts
    let hbestCost : bestCost ∈ costs := costs.min'_mem hcosts
    let winners := normalized.filter fun pair => mergeCost pair = bestCost
    let hwinners : winners.Nonempty := by
      rcases Finset.mem_image.mp hbestCost with ⟨pair, hpair, hpairCost⟩
      exact ⟨pair, Finset.mem_filter.mpr ⟨hpair, hpairCost⟩⟩
    let ranks := winners.image canonicalMergePairRank
    let hranks : ranks.Nonempty := Finset.image_nonempty.mpr hwinners
    let bestRank := ranks.min' hranks
    let hbestRank : bestRank ∈ ranks := ranks.min'_mem hranks
    some (Finset.mem_image.mp hbestRank).choose
  else
    none

theorem canonicalWardStep_eq_of_cost_eq
    {E : ℕ} (pairs : Finset (WardMergePair E))
    (cost₁ cost₂ : WardMergePair E → ℝ)
    (hcost : ∀ pair, cost₁ pair = cost₂ pair) :
    canonicalWardStep pairs cost₁ = canonicalWardStep pairs cost₂ := by
  have hfun : cost₁ = cost₂ := funext hcost
  subst cost₂
  rfl

/-- Apply a selected Ward merge to the current pool. -/
def applyWardMerge {E : ℕ} (pool : Finset (WardCluster E)) (pair : WardMergePair E) :
    Finset (WardCluster E) :=
  insert (pair.left ∪ pair.right) ((pool.erase pair.left).erase pair.right)

/-- Fuel-bounded canonical Ward run, stopping safely if no valid merge pair remains. -/
noncomputable def canonicalWardRun {E : ℕ} :
    ℕ → Finset (WardCluster E) →
      (Finset (WardCluster E) → WardMergePair E → ℝ) → Finset (WardCluster E)
  | 0, pool, _ => pool
  | fuel + 1, pool, mergeCost =>
      match canonicalWardStep (wardMergePairs pool) (mergeCost pool) with
      | none => pool
      | some pair => canonicalWardRun fuel (applyWardMerge pool pair) mergeCost

theorem canonicalWardRun_eq_of_cost_eq
    {E : ℕ} (fuel : ℕ) (pool : Finset (WardCluster E))
    (cost₁ cost₂ : Finset (WardCluster E) → WardMergePair E → ℝ)
    (hcost : ∀ current pair, cost₁ current pair = cost₂ current pair) :
    canonicalWardRun fuel pool cost₁ = canonicalWardRun fuel pool cost₂ := by
  have hfun : cost₁ = cost₂ := by
    funext current pair
    exact hcost current pair
  subst cost₂
  rfl

end BiostatFormal.HighDimensional.Multiview
