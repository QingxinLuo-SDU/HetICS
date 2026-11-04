import Mathlib.Data.Real.Basic
import Mathlib.Data.List.FinRange
import Mathlib.Tactic.Tauto

/-! Strict sequential selection and the approved frozen-label availability interface.
Real scores represent values already accepted by the implementation's finite check.
This module does not verify Python floating point execution or fit/guard certificates. -/

namespace BinaryFormal.FiniteFreeze

noncomputable section

variable {α : Type*}

/-- Replace the incumbent only for a strictly smaller score. -/
def improve (score : α → ℝ) (best : Option α) (next : α) : Option α :=
  match best with
  | none => some next
  | some old => if score next < score old then some next else some old

def choose (score : α → ℝ) (xs : List α) : Option α :=
  xs.foldl (improve score) none

/-- Invalid directional records are skipped without changing encounter order. -/
def chooseOptional (score : α → ℝ) (xs : List (Option α)) : Option α :=
  choose score (xs.filterMap id)

theorem fold_some_exists (score : α → ℝ) (xs : List α) (a : α) :
    ∃ b, xs.foldl (improve score) (some a) = some b := by
  induction xs generalizing a with
  | nil => exact ⟨a, rfl⟩
  | cons x xs ih =>
    simp only [List.foldl_cons, improve]
    split <;> apply ih

theorem fold_selected_mem (score : α → ℝ) (xs : List α) (acc : Option α) {a : α}
    (h : xs.foldl (improve score) acc = some a) : a ∈ xs ∨ acc = some a := by
  induction xs generalizing acc with
  | nil => exact Or.inr h
  | cons x xs ih =>
    rcases ih (improve score acc x) h with hm | ha
    · exact Or.inl (List.mem_cons_of_mem x hm)
    · cases acc with
      | none =>
        have : x = a := Option.some.inj ha
        exact Or.inl (by simp [this])
      | some old =>
        simp only [improve] at ha
        split at ha
        · have : x = a := Option.some.inj ha
          exact Or.inl (by simp [this])
        · exact Or.inr ha

theorem selected_mem (score : α → ℝ) (xs : List α) {a : α}
    (h : choose score xs = some a) : a ∈ xs := by
  rcases fold_selected_mem score xs none h with hm | hn
  · exact hm
  · cases hn

theorem choose_none_iff (score : α → ℝ) (xs : List α) :
    choose score xs = none ↔ xs = [] := by
  cases xs with
  | nil => simp [choose]
  | cons x xs =>
    obtain ⟨b, hb⟩ := fold_some_exists score xs x
    simp [choose, List.foldl_cons, improve, hb]

theorem choose_exists_iff (score : α → ℝ) (xs : List α) :
    (∃ a, choose score xs = some a) ↔ xs ≠ [] := by
  cases h : choose score xs with
  | none => simp [(choose_none_iff score xs).mp h]
  | some a =>
    constructor
    · intro _ he
      subst xs
      simp [choose] at h
    · intro _
      exact ⟨a, rfl⟩

theorem fold_le_initial (score : α → ℝ) (xs : List α) (a b : α)
    (h : xs.foldl (improve score) (some a) = some b) : score b ≤ score a := by
  induction xs generalizing a with
  | nil =>
    have hab : a = b := Option.some.inj h
    subst b
    exact le_refl _
  | cons x xs ih =>
    simp only [List.foldl_cons, improve] at h
    split at h
    · exact le_trans (ih x h) (le_of_lt ‹score x < score a›)
    · exact ih a h

theorem fold_le_member (score : α → ℝ) (xs : List α) (acc : Option α) {b : α}
    (h : xs.foldl (improve score) acc = some b) :
    ∀ a ∈ xs, score b ≤ score a := by
  induction xs generalizing acc with
  | nil => simp
  | cons x xs ih =>
    intro a ha
    simp only [List.mem_cons] at ha
    rcases ha with rfl | ha
    · cases acc with
      | none => exact fold_le_initial score xs a b h
      | some old =>
        simp only [List.foldl_cons, improve] at h
        split at h
        · exact fold_le_initial score xs a b h
        · exact le_trans (fold_le_initial score xs old b h) (le_of_not_gt ‹¬ score a < score old›)
    · exact ih (improve score acc x) h a ha

theorem selected_minimum (score : α → ℝ) (xs : List α) {a : α}
    (h : choose score xs = some a) : ∀ b ∈ xs, score a ≤ score b :=
  fold_le_member score xs none h

theorem fold_preserves_first_minimum (score : α → ℝ) (xs : List α) (a : α)
    (h : ∀ b ∈ xs, score a ≤ score b) :
    xs.foldl (improve score) (some a) = some a := by
  induction xs with
  | nil => rfl
  | cons b xs ih =>
    have hab : ¬ score b < score a := not_lt_of_ge (h b (by simp))
    simpa [List.foldl_cons, improve, hab] using ih (fun c hc => h c (by simp [hc]))

/-- General first-minimum characterization: every earlier valid score is strictly larger,
and every later valid score is at least this score. Thus later ties cannot replace it. -/
theorem choose_first_minimum (score : α → ℝ) (pre post : List α) (a : α)
    (hpre : ∀ b ∈ pre, score a < score b)
    (hpost : ∀ b ∈ post, score a ≤ score b) :
    choose score (pre ++ a :: post) = some a := by
  simp only [choose, List.foldl_append, List.foldl_cons]
  cases hp : pre.foldl (improve score) none with
  | none => exact fold_preserves_first_minimum score post a hpost
  | some b =>
    have hb := selected_mem score pre hp
    have hab := hpre b hb
    simpa [improve, hab] using fold_preserves_first_minimum score post a hpost

theorem optional_selected_valid (score : α → ℝ) (xs : List (Option α)) {a : α}
    (h : chooseOptional score xs = some a) : some a ∈ xs := by
  have := selected_mem score (xs.filterMap id) h
  simpa using this

theorem optional_none_iff (score : α → ℝ) (xs : List (Option α)) :
    chooseOptional score xs = none ↔ ∀ a, some a ∉ xs := by
  rw [chooseOptional, choose_none_iff]
  simp [List.eq_nil_iff_forall_not_mem]

theorem optional_minimum (score : α → ℝ) (xs : List (Option α)) {a : α}
    (h : chooseOptional score xs = some a) :
    ∀ b, some b ∈ xs → score a ≤ score b := by
  intro b hb
  exact selected_minimum score (xs.filterMap id) h b (by simpa using hb)

theorem optional_exists_iff (score : α → ℝ) (xs : List (Option α)) :
    (∃ a, chooseOptional score xs = some a) ↔ ∃ a, some a ∈ xs := by
  constructor
  · rintro ⟨a, ha⟩
    exact ⟨a, optional_selected_valid score xs ha⟩
  · rintro ⟨a, ha⟩
    cases h : chooseOptional score xs with
    | none => exact ((optional_none_iff score xs).mp h a ha).elim
    | some b => exact ⟨b, rfl⟩

theorem improve_map {β : Type*} (s : α → ℝ) (t : β → ℝ) (f : α → β)
    (h : ∀ a, t (f a) = s a) (acc : Option α) (a : α) :
    improve t (acc.map f) (f a) = (improve s acc a).map f := by
  cases acc with
  | none => rfl
  | some b =>
    simp only [Option.map_some, improve, h]
    split <;> rfl

theorem fold_map {β : Type*} (s : α → ℝ) (t : β → ℝ) (f : α → β)
    (h : ∀ a, t (f a) = s a) (xs : List α) (acc : Option α) :
    (xs.map f).foldl (improve t) (acc.map f) = (xs.foldl (improve s) acc).map f := by
  induction xs generalizing acc with
  | nil => rfl
  | cons x xs ih =>
    simpa only [List.map_cons, List.foldl_cons, improve_map s t f h] using
      ih (improve s acc x)

/-- Project the actually scored fold to the old candidate interface when the recorded
finite values equal that interface's empirical risk. This is score-data agreement only. -/
theorem choose_scored_projection (score : α → ℝ) (xs : List (α × ℝ))
    (h : ∀ p ∈ xs, p.2 = score p.1) :
    (choose Prod.snd xs).map Prod.fst = choose score (xs.map Prod.fst) := by
  let tag : α → α × ℝ := fun a => (a, score a)
  have hx : (xs.map Prod.fst).map tag = xs := by
    rw [List.map_map]
    calc
      xs.map (tag ∘ Prod.fst) = xs.map id := by
        apply List.map_congr_left
        intro p hp
        exact Prod.ext rfl (h p hp).symm
      _ = xs := List.map_id xs
  have hf := fold_map score Prod.snd tag (fun _ => rfl) (xs.map Prod.fst) none
  change choose Prod.snd ((xs.map Prod.fst).map tag) =
    (choose score (xs.map Prod.fst)).map tag at hf
  rw [hx] at hf
  rw [hf, Option.map_map]
  cases choose score (xs.map Prod.fst) <;> rfl

/-- A missing/nonfinite/ineligible risk is None. Both sides must exist and have permitted
families before either direction is submitted. These are raw input facts, not certificates. -/
structure RawDirection (α : Type*) where
  candidate : α
  risk : Option ℝ
  familyAllowed : Bool

def scoredDirection (r : RawDirection α) : Option (α × ℝ) :=
  r.risk.map (fun risk => (r.candidate, risk))

def logicalPair (left right : Option (RawDirection α)) : List (Option (α × ℝ)) :=
  match left, right with
  | some l, some r =>
      if l.familyAllowed && r.familyAllowed then [scoredDirection l, scoredDirection r] else []
  | _, _ => []

def primaryChoice (logicalOrder : List (Option (RawDirection α) × Option (RawDirection α))) :
    Option (α × ℝ) :=
  chooseOptional Prod.snd (logicalOrder.flatMap fun p => logicalPair p.1 p.2)

/-- Refit data may report changed labels, missing parameters, or ineligibility. -/
structure Refit (Center Label Parameter : Type*) where
  postLabel : Center → Label
  eligible : Bool
  parameters : Option Parameter
  converged : Bool

inductive Completion where
  | complete
  | completeMaxIterEligible
  deriving DecidableEq

/-- The outer option is selection; diagnostics survive unsuccessful refits. -/
structure FrozenResult (Center Label Parameter : Type*) where
  frozenLabel : Center → Label
  downstream : Option (Parameter × Completion)

def usableRefit {Center Label Parameter : Type*} (frozen : Center → Label)
    (refit : Option (Refit Center Label Parameter)) : Option (Parameter × Completion) := by
  classical
  exact match refit with
  | none => none
  | some r =>
      if r.eligible = true ∧ r.postLabel = frozen then
        r.parameters.map (fun p => (p, if r.converged then .complete else .completeMaxIterEligible))
      else none

def freeze {Center Label Parameter : Type*} (selection : Option (Center → Label))
    (refit : Option (Refit Center Label Parameter)) : Option (FrozenResult Center Label Parameter) :=
  selection.map (fun frozen => ⟨frozen, usableRefit frozen refit⟩)

theorem freeze_abstain {Center Label Parameter : Type*}
    (refit : Option (Refit Center Label Parameter)) : freeze none refit = none := rfl

theorem freeze_retains_labels {Center Label Parameter : Type*} (frozen : Center → Label)
    (refit : Option (Refit Center Label Parameter)) :
    (freeze (some frozen) refit).map FrozenResult.frozenLabel = some frozen := rfl

theorem usableRefit_iff {Center Label Parameter : Type*} (frozen : Center → Label)
    (refit : Option (Refit Center Label Parameter)) :
    (∃ p status, usableRefit frozen refit = some (p, status)) ↔
      ∃ r p, refit = some r ∧ r.eligible = true ∧ r.postLabel = frozen ∧ r.parameters = some p := by
  classical
  cases refit with
  | none => simp [usableRefit]
  | some r =>
    by_cases h : r.eligible = true ∧ r.postLabel = frozen
    · simp only [usableRefit, h, Option.some.injEq]
      cases hp : r.parameters <;> simp_all
    · simp [usableRefit, h]
      tauto

theorem usableRefit_missing {Center Label Parameter : Type*} (frozen : Center → Label) :
    usableRefit (Parameter := Parameter) frozen none = none := rfl

theorem usableRefit_changed {Center Label Parameter : Type*} (frozen : Center → Label)
    (r : Refit Center Label Parameter) (h : r.postLabel ≠ frozen) :
    usableRefit frozen (some r) = none := by simp [usableRefit, h]

theorem usableRefit_ineligible {Center Label Parameter : Type*} (frozen : Center → Label)
    (r : Refit Center Label Parameter) (h : r.eligible = false) :
    usableRefit frozen (some r) = none := by simp [usableRefit, h]

theorem usableRefit_no_parameters {Center Label Parameter : Type*} (frozen : Center → Label)
    (r : Refit Center Label Parameter) (h : r.parameters = none) :
    usableRefit frozen (some r) = none := by simp [usableRefit, h]

theorem usableRefit_max_iter {Center Label Parameter : Type*} (frozen : Center → Label)
    (r : Refit Center Label Parameter) (p : Parameter)
    (he : r.eligible = true) (hl : r.postLabel = frozen)
    (hp : r.parameters = some p) (hc : r.converged = false) :
    usableRefit frozen (some r) = some (p, .completeMaxIterEligible) := by
  simp [usableRefit, he, hl, hp, hc]

def rowLabel {Row Center Label : Type*} (center : Row → Center)
    (frozen : Center → Label) : Row → Label := frozen ∘ center

theorem same_center_same_label {Row Center Label : Type*} (center : Row → Center)
    (frozen : Center → Label) {i j : Row} (h : center i = center j) :
    rowLabel center frozen i = rowLabel center frozen j := congrArg frozen h

theorem injective_relabel_preserves_cluster {Row Center Label NewLabel : Type*}
    (center : Row → Center) (frozen : Center → Label) (rename : Label → NewLabel)
    (h : Function.Injective rename) (i j : Row) :
    rowLabel center (rename ∘ frozen) i = rowLabel center (rename ∘ frozen) j ↔
      rowLabel center frozen i = rowLabel center frozen j := h.eq_iff

end

#print axioms improve
#print axioms choose
#print axioms chooseOptional
#print axioms fold_some_exists
#print axioms fold_selected_mem
#print axioms selected_mem
#print axioms choose_none_iff
#print axioms choose_exists_iff
#print axioms fold_le_initial
#print axioms fold_le_member
#print axioms selected_minimum
#print axioms fold_preserves_first_minimum
#print axioms choose_first_minimum
#print axioms optional_selected_valid
#print axioms optional_none_iff
#print axioms optional_minimum
#print axioms optional_exists_iff
#print axioms improve_map
#print axioms fold_map
#print axioms choose_scored_projection
#print axioms RawDirection
#print axioms scoredDirection
#print axioms logicalPair
#print axioms primaryChoice
#print axioms Refit
#print axioms Completion
#print axioms FrozenResult
#print axioms usableRefit
#print axioms freeze
#print axioms freeze_abstain
#print axioms freeze_retains_labels
#print axioms usableRefit_iff
#print axioms usableRefit_missing
#print axioms usableRefit_changed
#print axioms usableRefit_ineligible
#print axioms usableRefit_no_parameters
#print axioms usableRefit_max_iter
#print axioms rowLabel
#print axioms same_center_same_label
#print axioms injective_relabel_preserves_cluster

end BinaryFormal.FiniteFreeze
