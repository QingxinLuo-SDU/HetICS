import FiniteFreeze
import BiostatFormal.HighDimensional.Multiview.SelectorContract

/-! Consumers of strict encounter-order selection, including the existing B1 interface.
The adapter uses the identical empirical risk function for selection and the contract.
It does not assume a selector contract, totality, optimality, or recovery as an input. -/

namespace BinaryFormal.FiniteFreeze

noncomputable section

open BiostatFormal.HighDimensional.Multiview

variable {Candidate Omega EvaluationContext : Type*}
variable [DecidableEq Candidate] [RiskInput Candidate]

def orderedPool (xs : List (Option Candidate)) : GuardedCandidatePool Candidate :=
  (xs.filterMap id).toFinset

def orderedOutput (score : Candidate → ℝ) (xs : List (Option Candidate)) :
    AlgorithmOutput (orderedPool xs) SelectorAbstainReason :=
  match h : chooseOptional score xs with
  | none => .abstain .emptyPool
  | some a => .selected ⟨a, by
      have ha := optional_selected_valid score xs h
      simpa [orderedPool] using ha⟩

omit [RiskInput Candidate] in
theorem orderedOutput_selected (score : Candidate → ℝ) (xs : List (Option Candidate))
    (a : GuardedPoolMember (orderedPool xs)) :
    orderedOutput score xs = .selected a ↔ chooseOptional score xs = some a.1 := by
  unfold orderedOutput
  split <;> simp_all [Subtype.ext_iff]

theorem ordered_contract (ordered : Omega → List (Option Candidate))
    (spec : SelectorRiskSpec Candidate EvaluationContext) :
    SelectorContract (fun ω => orderedPool (ordered ω)) spec
      (fun ω => orderedOutput (empiricalCompositeRisk spec.risk) (ordered ω)) 0 0 0 := by
  constructor
  · intro ω hn
    obtain ⟨a, ha⟩ := hn
    have hm : some a ∈ ordered ω := by simpa [orderedPool] using ha
    have hne : chooseOptional (empiricalCompositeRisk spec.risk) (ordered ω) ≠ none := by
      intro he
      exact (optional_none_iff _ _).mp he a hm
    cases h : chooseOptional (empiricalCompositeRisk spec.risk) (ordered ω) with
    | none => exact (hne h).elim
    | some b =>
      have hb := optional_selected_valid _ _ h
      refine ⟨⟨b, by simpa [orderedPool] using hb⟩, ?_⟩
      exact (orderedOutput_selected _ _ _).mpr h
  · intro ω selected hs candidate
    have hs' := (orderedOutput_selected _ _ _).mp hs
    have hm : some candidate.1 ∈ ordered ω := by
      simpa only [orderedPool, List.mem_toFinset, List.mem_filterMap, id_eq,
        exists_eq_right] using candidate.2
    have hmin := optional_minimum _ _ hs' candidate.1 hm
    simpa [selectionTolerance] using hmin

theorem ordered_contract_nonnegative_tolerance (ordered : Omega → List (Option Candidate))
    (spec : SelectorRiskSpec Candidate EvaluationContext) (ηt ηg ηn : ℝ)
    (hη : 0 ≤ selectionTolerance ηt ηg ηn) :
    SelectorContract (fun ω => orderedPool (ordered ω)) spec
      (fun ω => orderedOutput (empiricalCompositeRisk spec.risk) (ordered ω)) ηt ηg ηn := by
  have hc := ordered_contract ordered spec
  constructor
  · exact hc.selects_of_nonempty
  · intro ω selected hs candidate
    have hm := hc.nearOptimal_of_selected ω selected hs candidate
    simp only [selectionTolerance, add_zero] at hm
    exact le_trans hm (le_add_of_nonneg_right hη)

omit [RiskInput Candidate] in
/-- Agreement of the selected constructor with the actual scored fold. A risk-value equality
is an input fact connecting fields; optimality and totality remain proved by the fold. -/
theorem scored_fold_agrees_with_output (score : Candidate → ℝ) (xs : List (Candidate × ℝ))
    (hscore : ∀ p ∈ xs, p.2 = score p.1)
    (selected : GuardedPoolMember (orderedPool (xs.map fun p => some p.1))) :
    orderedOutput score (xs.map fun p => some p.1) = .selected selected ↔
      (choose Prod.snd xs).map Prod.fst = some selected.1 := by
  rw [orderedOutput_selected, choose_scored_projection score xs hscore]
  simp [chooseOptional, List.filterMap_map]

/-- The same valid encounter order used by primaryChoice, projected to contextual records. -/
def primaryOrder (logical : List (Option (RawDirection Candidate) × Option (RawDirection Candidate))) :
    List (Option Candidate) :=
  ((logical.flatMap fun p => logicalPair p.1 p.2).filterMap id).map (fun p => some p.1)

omit [RiskInput Candidate] in
theorem primary_fold_agrees_with_output (score : Candidate → ℝ)
    (logical : List (Option (RawDirection Candidate) × Option (RawDirection Candidate)))
    (hscore : ∀ p ∈ (logical.flatMap fun p => logicalPair p.1 p.2).filterMap id,
      p.2 = score p.1)
    (selected : GuardedPoolMember (orderedPool (primaryOrder logical))) :
    orderedOutput score (primaryOrder logical) = .selected selected ↔
      (primaryChoice logical).map Prod.fst = some selected.1 :=
  scored_fold_agrees_with_output score _ hscore selected

/-- Exact survival forces nonempty input; the fold supplies selected near-optimality.
No assumption on the success event is used. -/
theorem ordered_selector_success_univ
    {Center Label Slot RecordContext EvaluationContext Omega : Type*}
    [DecidableEq (CandidateRecord Center Label Slot RecordContext)]
    (ordered : Omega → List (Option (CandidateRecord Center Label Slot RecordContext)))
    (spec : SelectorRiskSpec (CandidateRecord Center Label Slot RecordContext) EvaluationContext)
    (truth : Partition Center Label) :
    selectorSuccessEvent (fun ω => orderedPool (ordered ω)) spec
      (fun ω => orderedOutput (empiricalCompositeRisk spec.risk) (ordered ω)) truth 0 0 0 =
      Set.univ := by
  classical
  have hc := ordered_contract ordered spec
  apply Set.eq_univ_of_forall
  intro ω
  by_cases hs : ω ∈ exactSurvivalEvent (fun ω => orderedPool (ordered ω)) truth
  · obtain ⟨z, _⟩ := hs
    obtain ⟨selected, ho⟩ := hc.selects_of_nonempty ω ⟨z.1, z.2⟩
    exact Or.inr ⟨selected, ho, hc.nearOptimal_of_selected ω selected ho⟩
  · exact Or.inl hs

end

noncomputable section

variable {Candidate : Type*}

/-- Within one logical candidate, AB precedes BA and wins an exact score tie. -/
theorem ab_precedes_ba_on_tie (ab ba : Candidate) (score : Candidate → ℝ)
    (h : score ab = score ba) :
    chooseOptional score [some ab, some ba] = some ab := by
  apply choose_first_minimum score [] [ba] ab
  · simp
  · intro b hb
    simp only [List.mem_singleton] at hb
    subst b
    exact le_of_eq h

theorem invalid_directions_skipped (a : Candidate) (score : Candidate → ℝ) :
    chooseOptional score [none, some a, none] = some a := by
  simp [chooseOptional, choose, improve]

theorem later_strict_improvement (a b : Candidate) (score : Candidate → ℝ)
    (h : score b < score a) :
    chooseOptional score [some a, some b] = some b := by
  simp [chooseOptional, choose, improve, h]

theorem all_invalid_abstains (score : Candidate → ℝ) :
    chooseOptional score [none, none, none] = none := rfl

theorem missing_other_side_excludes (r : RawDirection Candidate) :
    logicalPair (some r) none = [] ∧ logicalPair none (some r) = [] := ⟨rfl, rfl⟩

theorem family_filter_excludes (left right : RawDirection Candidate)
    (h : left.familyAllowed = false) : logicalPair (some left) (some right) = [] := by
  simp [logicalPair, h]

theorem one_valid_direction_suffices (ab ba : Candidate) (r : ℝ) :
    primaryChoice [(some ⟨ab, some r, true⟩, some ⟨ba, none, true⟩)] = some (ab, r) := by
  simp [primaryChoice, logicalPair, scoredDirection, chooseOptional, choose, improve]

theorem primary_tie_ab (ab ba : Candidate) (r : ℝ) :
    primaryChoice [(some ⟨ab, some r, true⟩, some ⟨ba, some r, true⟩)] = some (ab, r) := by
  simp [primaryChoice, logicalPair, scoredDirection, chooseOptional, choose, improve]

theorem refit_failure_keeps_diagnostics {Center Label Parameter : Type*} (frozen : Center → Label)
    (r : Refit Center Label Parameter) (h : r.eligible = false) :
    freeze (some frozen) (some r) = some ⟨frozen, none⟩ := by
  simp [freeze, usableRefit_ineligible frozen r h]

theorem max_iter_is_usable {Center Label Parameter : Type*} (frozen : Center → Label)
    (p : Parameter) :
    freeze (some frozen) (some ⟨frozen, true, some p, false⟩) =
      some ⟨frozen, some (p, .completeMaxIterEligible)⟩ := by
  simp [freeze, usableRefit]

end

end BinaryFormal.FiniteFreeze

#print axioms BinaryFormal.FiniteFreeze.orderedPool
#print axioms BinaryFormal.FiniteFreeze.orderedOutput
#print axioms BinaryFormal.FiniteFreeze.orderedOutput_selected
#print axioms BinaryFormal.FiniteFreeze.ordered_contract
#print axioms BinaryFormal.FiniteFreeze.ordered_contract_nonnegative_tolerance
#print axioms BinaryFormal.FiniteFreeze.scored_fold_agrees_with_output
#print axioms BinaryFormal.FiniteFreeze.primaryOrder
#print axioms BinaryFormal.FiniteFreeze.primary_fold_agrees_with_output
#print axioms BinaryFormal.FiniteFreeze.ordered_selector_success_univ
#print axioms BinaryFormal.FiniteFreeze.ab_precedes_ba_on_tie
#print axioms BinaryFormal.FiniteFreeze.invalid_directions_skipped
#print axioms BinaryFormal.FiniteFreeze.later_strict_improvement
#print axioms BinaryFormal.FiniteFreeze.all_invalid_abstains
#print axioms BinaryFormal.FiniteFreeze.missing_other_side_excludes
#print axioms BinaryFormal.FiniteFreeze.family_filter_excludes
#print axioms BinaryFormal.FiniteFreeze.one_valid_direction_suffices
#print axioms BinaryFormal.FiniteFreeze.primary_tie_ab
#print axioms BinaryFormal.FiniteFreeze.refit_failure_keeps_diagnostics
#print axioms BinaryFormal.FiniteFreeze.max_iter_is_usable
