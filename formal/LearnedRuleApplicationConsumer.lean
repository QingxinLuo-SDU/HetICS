import LearnedRuleApplication

noncomputable section
open Filter MeasureTheory
open scoped Topology
namespace BinaryFormal.LearnedRuleApplicationConsumer
open BiostatFormal.HighDimensional.Multiview
open FiniteFreeze B1Current LearnedRuleApplication

section B1
variable {Ω Context Center Label Slot RC EC V Parameter : Type*}
  [Fintype Center] [DecidableEq Center] [Fintype Label] [DecidableEq Label]
  [DecidableEq (CandidateRecord Center Label Slot RC)] [DecidableEq V]

/-- Real calls to ordered recovery and frozen exact selection establish L; complete
candidate/official-ICP stability and lower-level usability then establish A=C. -/
theorem b1_to_complete_report
    (ordered : Ω → List (Option (CandidateRecord Center Label Slot RC)))
    (spec : SelectorRiskSpec (CandidateRecord Center Label Slot RC) EC)
    (truth : Partition Center Label) (u : ℝ) (ω : Ω)
    (hB1 : ω ∈ exactSurvivalEvent (currentPool ordered) truth ∩
      uniformDeviationEvent (currentPool ordered) spec u ∩
      selectorStrictPopulationGapEvent (currentPool ordered) spec truth u 0 0 0)
    (pipe : Pipeline Context Center V Parameter) (ctxA ctxR : Context)
    (p : Parameter) (S0 : Center → Finset V) (refit : Option (Refit Center Label Parameter))
    (hS : stableEvent pipe ctxR truth S0)
    (hU : usableEvent pipe ctxA ctxR p truth S0 (frozenOutput (ordered ω) spec refit)) :
    actual pipe ctxA (frozenOutput (ordered ω) spec refit) = reference pipe ctxR p truth S0 ∧
    actual pipe ctxA (frozenOutput (ordered ω) spec refit) ≠ none := by
  have recovered := ordered_recovery ordered spec truth u hB1
  obtain ⟨frozen, hf, _, hmembers⟩ := frozen_selected_exact ordered spec truth refit recovered
  exact fields_to_report pipe ctxA ctxR p truth S0 _
    ⟨⟨frozen, usableRefit frozen refit⟩, hf, hmembers⟩ hS hU

/-- The complete probabilistic consumer. L may be the event on which B1's source,
deviation and gap premises hold; its failure budget already includes all B1 guards.
No samples are removed, and the reference sets S0 n do not depend on omega. -/
theorem b1_to_two_scales [MeasurableSpace Ω] (μ : Measure Ω)
    (ordered : ℕ → Ω → List (Option (CandidateRecord Center Label Slot RC)))
    (spec : ℕ → SelectorRiskSpec (CandidateRecord Center Label Slot RC) EC)
    (truth : Partition Center Label) (u : ℕ → ℝ)
    (pipe : Pipeline Context Center V Parameter) (ctxA ctxR : ℕ → Ω → Context)
    (p : ℕ → Ω → Parameter) (S0 : ℕ → Center → Finset V)
    (refit : ℕ → Ω → Option (Refit Center Label Parameter))
    (L S U : ℕ → Set Ω) (δL δS δU : ℕ → ENNReal)
    (hB1 : ∀ n ω, ω ∈ L n → ω ∈
      exactSurvivalEvent (currentPool (ordered n)) truth ∩
      uniformDeviationEvent (currentPool (ordered n)) (spec n) (u n) ∩
      selectorStrictPopulationGapEvent (currentPool (ordered n)) (spec n) truth (u n) 0 0 0)
    (hS : ∀ n ω, ω ∈ S n → stableEvent pipe (ctxR n ω) truth (S0 n))
    (hU : ∀ n ω, ω ∈ U n → usableEvent pipe (ctxA n ω) (ctxR n ω) (p n ω) truth (S0 n)
      (frozenOutput (ordered n ω) (spec n) (refit n ω)))
    (bounds : ∀ n, μ (L n)ᶜ ≤ δL n ∧ μ (S n)ᶜ ≤ δS n ∧ μ (U n)ᶜ ≤ δU n)
    (hδL : Tendsto δL atTop (𝓝 0)) (hδS : Tendsto δS atTop (𝓝 0))
    (hδU : Tendsto δU atTop (𝓝 0)) (fallbackA fallbackC : ℕ → Ω → Report) :
    let A := fun n ω => actual pipe (ctxA n ω) (frozenOutput (ordered n ω) (spec n) (refit n ω))
    let C := fun n ω => reference pipe (ctxR n ω) (p n ω) truth (S0 n)
    TendstoInMeasure μ (fun (n : ℕ) ω => Real.sqrt ((n : ℝ) + 1) *
      ((extend (fallbackA n ω) (A n ω)).point - (extend (fallbackC n ω) (C n ω)).point)) atTop 0 ∧
    TendstoInMeasure μ (fun (n : ℕ) ω => ((n : ℝ) + 1) *
      ((extend (fallbackA n ω) (A n ω)).variance - (extend (fallbackC n ω) (C n ω)).variance)) atTop 0 ∧
    Tendsto (fun n => μ {ω | A n ω = none}) atTop (𝓝 0) := by
  dsimp only
  apply reports_to_scaled_differences μ _ _ fallbackA fallbackC _ _ (fun n => L n ∩ S n ∩ U n)
  · intro n ω hw
    exact b1_to_complete_report (ordered n) (spec n) truth (u n) ω (hB1 n ω hw.1.1)
      pipe (ctxA n ω) (ctxR n ω) (p n ω) (S0 n) (refit n ω) (hS n ω hw.1.2) (hU n ω hw.2)
  · exact three_failure_tendsto μ L S U δL δS δU bounds hδL hδS hδU
end B1

def finitePipe : Pipeline Unit Bool ℕ Unit where
  channel := .dual
  candidateX := fun _ _ _ => {0}
  candidateY := fun _ _ _ => {1}
  officialICP := fun _ _ _ s => s.erase 1
  fit := fun _ _ _ s => some ⟨((s true).card : ℝ), 1⟩

/-- The vectors (false,true) and (true,false) differ at every entry but give the same report. -/
theorem swapped_labels_same_report :
    learned finitePipe () () (fun b : Bool => !b) = learned finitePipe () () id := by
  apply learned_relabel
  intro i j
  cases i <;> cases j <;> decide

/-- Same labels and fit context, different final sets: the report is different. -/
theorem same_labels_different_final_sets :
    reference finitePipe () () (id : Bool → Bool) (fun _ => {0}) ≠
      reference finitePipe () () (id : Bool → Bool) (fun _ => ∅) := by
  simp [reference, finitePipe]

theorem official_icp_is_consumed : finalSets finitePipe () (members (id : Bool → Bool)) =
    fun _ => {0} := by
  funext c
  simp [finalSets, finitePipe, ScreeningAdjustment.finalSelection, ScreeningAdjustment.candidates]
  decide

/-- A reference fit may exist while actual refit refusal remains a recorded failure. -/
theorem rejected_actual_with_available_reference :
    actual finitePipe () (some ⟨(id : Bool → Bool), none⟩) = none ∧
    reference finitePipe () () (id : Bool → Bool) (fun _ => {0}) = some ⟨1, 1⟩ := by
  constructor
  · exact actual_rejected finitePipe () id
  · simp [reference, finitePipe]

/-- Nonconverged, eligible, unchanged labels and present parameters retain permission. -/
theorem max_iter_remains_eligible :
    actual finitePipe () (freeze (some (id : Bool → Bool))
      (some ⟨id, true, some (), false⟩)) = some ⟨1, 1⟩ := by
  have h := usableRefit_max_iter (id : Bool → Bool)
    (⟨id, true, some (), false⟩ : Refit Bool Bool Unit) () rfl rfl rfl rfl
  simp only [freeze, Option.map_some, actual, Option.bind_some, h]
  change learned finitePipe () () (id : Bool → Bool) = some ⟨1, 1⟩
  unfold learned
  rw [official_icp_is_consumed]
  simp [finitePipe]

#print axioms b1_to_complete_report
#print axioms b1_to_two_scales
#print axioms finitePipe
#print axioms swapped_labels_same_report
#print axioms same_labels_different_final_sets
#print axioms official_icp_is_consumed
#print axioms rejected_actual_with_available_reference
#print axioms max_iter_remains_eligible
end BinaryFormal.LearnedRuleApplicationConsumer
