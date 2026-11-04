import B1Current
import ScreeningAdjustment
import ErrorTarget
import FixedCenterM

/-! Full-data rule transport. Candidates include the entire registered channel and
the official ICP call. Statistical stability is an explicit additional condition. -/
noncomputable section
open Filter MeasureTheory
open scoped Topology
namespace BinaryFormal.LearnedRuleApplication
open FiniteFreeze ScreeningAdjustment

structure Report where
  point : ℝ
  variance : ℝ
  deriving DecidableEq

def members {Center Label : Type*} (label : Center → Label) : Center → Set Center :=
  fun c => {d | label d = label c}

theorem members_eq {Center Label : Type*} (a b : Center → Label)
    (h : ∀ i j, a i = a j ↔ b i = b j) : members a = members b := by
  funext c
  ext d
  exact h d c

structure Pipeline (Context Center V Parameter : Type*) [DecidableEq V] where
  channel : Channel
  candidateX : Context → (Center → Set Center) → Center → Finset V
  candidateY : Context → (Center → Set Center) → Center → Finset V
  officialICP : Context → (Center → Set Center) → Center → Finset V → Finset V
  fit : Context → Parameter → (Center → Set Center) → (Center → Finset V) → Option Report

variable {Context Center Label V Parameter : Type*} [DecidableEq V]

def finalSets (pipe : Pipeline Context Center V Parameter) (ctx : Context)
    (m : Center → Set Center) : Center → Finset V := fun c =>
  finalSelection (pipe.officialICP ctx m c) pipe.channel
    (pipe.candidateX ctx m c) (pipe.candidateY ctx m c)

/-- B: truth labels, still running the complete random screening rule. -/
def learned (pipe : Pipeline Context Center V Parameter) (ctx : Context)
    (p : Parameter) (label : Center → Label) : Option Report :=
  pipe.fit ctx p (members label) (finalSets pipe ctx (members label))

/-- C: a fixed set for each center, with the same reference parameter fitting rule. -/
def reference (pipe : Pipeline Context Center V Parameter) (ctx : Context)
    (p : Parameter) (truth : Center → Label) (S0 : Center → Finset V) : Option Report :=
  pipe.fit ctx p (members truth) S0

/-- A: unsuccessful selection/refit remains none, including changed frozen labels.
Completion tags are retained upstream; max_iter_eligible is permitted here. -/
def actual (pipe : Pipeline Context Center V Parameter) (ctx : Context)
    (state : Option (FrozenResult Center Label Parameter)) : Option Report :=
  state.bind fun fr => fr.downstream.bind fun ps => learned pipe ctx ps.1 fr.frozenLabel

theorem learned_relabel (pipe : Pipeline Context Center V Parameter) (ctx : Context)
    (p : Parameter) (a b : Center → Label)
    (h : ∀ i j, a i = a j ↔ b i = b j) :
    learned pipe ctx p a = learned pipe ctx p b := by
  unfold learned
  rw [members_eq a b h]

theorem learned_eq_reference (pipe : Pipeline Context Center V Parameter) (ctx : Context)
    (p : Parameter) (truth : Center → Label) (S0 : Center → Finset V)
    (stable : finalSets pipe ctx (members truth) = S0) :
    learned pipe ctx p truth = reference pipe ctx p truth S0 := by
  simp only [learned, reference, stable]

theorem actual_eq_reference (pipe : Pipeline Context Center V Parameter)
    (ctxA ctxR : Context) (p : Parameter) (a truth : Center → Label)
    (S0 : Center → Finset V) (status : Completion)
    (settings : ctxA = ctxR) (labels : ∀ i j, a i = a j ↔ truth i = truth j)
    (stable : finalSets pipe ctxR (members truth) = S0) :
    actual pipe ctxA (some ⟨a, some (p, status)⟩) = reference pipe ctxR p truth S0 := by
  subst ctxA
  change learned pipe ctxR p a = _
  exact (learned_relabel pipe ctxR p a truth labels).trans
    (learned_eq_reference pipe ctxR p truth S0 stable)

theorem actual_rejected (pipe : Pipeline Context Center V Parameter) (ctx : Context)
    (a : Center → Label) : actual pipe ctx (some ⟨a, none⟩) = none := rfl

def labelEvent (state : Option (FrozenResult Center Label Parameter))
    (truth : Center → Label) : Prop :=
  ∃ fr, state = some fr ∧ ∀ i j, fr.frozenLabel i = fr.frozenLabel j ↔ truth i = truth j

def stableEvent (pipe : Pipeline Context Center V Parameter) (ctx : Context)
    (truth : Center → Label) (S0 : Center → Finset V) : Prop :=
  finalSets pipe ctx (members truth) = S0

/-- Parameter agreement is a lower-level input in addition to data/seed/order/settings.
It must not be inferred from recovery of labels. -/
def usableEvent (pipe : Pipeline Context Center V Parameter) (ctxA ctxR : Context)
    (p : Parameter) (truth : Center → Label) (S0 : Center → Finset V)
    (state : Option (FrozenResult Center Label Parameter)) : Prop :=
  ctxA = ctxR ∧ (∃ r, reference pipe ctxR p truth S0 = some r) ∧
    ∀ fr, state = some fr → ∃ status, fr.downstream = some (p, status)

theorem fields_to_report (pipe : Pipeline Context Center V Parameter) (ctxA ctxR : Context)
    (p : Parameter) (truth : Center → Label) (S0 : Center → Finset V)
    (state : Option (FrozenResult Center Label Parameter))
    (hL : labelEvent state truth) (hS : stableEvent pipe ctxR truth S0)
    (hU : usableEvent pipe ctxA ctxR p truth S0 state) :
    actual pipe ctxA state = reference pipe ctxR p truth S0 ∧ actual pipe ctxA state ≠ none := by
  obtain ⟨fr, hs, hl⟩ := hL
  obtain ⟨hc, ⟨r, hr⟩, hu⟩ := hU
  obtain ⟨status, hp⟩ := hu fr hs
  have he : actual pipe ctxA state = reference pipe ctxR p truth S0 := by
    rw [hs]
    cases fr with
    | mk a downstream =>
      dsimp only at hp hl
      rw [hp]
      exact actual_eq_reference pipe ctxA ctxR p a truth S0 status hc hl hS
  exact ⟨he, by rw [he, hr]; exact Option.some_ne_none r⟩

/-- A totalization for limit arguments only. Actual availability is still the Option.
Different arbitrary extensions may be used for A and C. -/
def extend (fallback : Report) (r : Option Report) : Report := r.getD fallback

section Probability
variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)

theorem three_failure_bound (L S U : Set Ω) (δL δS δU : ENNReal)
    (hL : μ Lᶜ ≤ δL) (hS : μ Sᶜ ≤ δS) (hU : μ Uᶜ ≤ δU) :
    μ (L ∩ S ∩ U)ᶜ ≤ δL + δS + δU := by
  rw [Set.compl_inter, Set.compl_inter]
  exact (measure_union_le _ _).trans
    (add_le_add ((measure_union_le _ _).trans (add_le_add hL hS)) hU)

theorem three_failure_tendsto (L S U : ℕ → Set Ω) (δL δS δU : ℕ → ENNReal)
    (bounds : ∀ n, μ (L n)ᶜ ≤ δL n ∧ μ (S n)ᶜ ≤ δS n ∧ μ (U n)ᶜ ≤ δU n)
    (hL : Tendsto δL atTop (𝓝 0)) (hS : Tendsto δS atTop (𝓝 0))
    (hU : Tendsto δU atTop (𝓝 0)) :
    Tendsto (fun n => μ (L n ∩ S n ∩ U n)ᶜ) atTop (𝓝 0) := by
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
    (by simpa using (hL.add hS).add hU)
  · intro n; exact bot_le
  · intro n
    exact three_failure_bound μ _ _ _ _ _ _ (bounds n).1 (bounds n).2.1 (bounds n).2.2

/-- Complete Option disagreement OR actual unavailability is budgeted. -/
theorem report_failure_bound (A C : Ω → Option Report) (L S U : Set Ω)
    (fields : ∀ ω, ω ∈ L ∩ S ∩ U → A ω = C ω ∧ A ω ≠ none)
    (δL δS δU : ENNReal) (hL : μ Lᶜ ≤ δL) (hS : μ Sᶜ ≤ δS) (hU : μ Uᶜ ≤ δU) :
    μ {ω | A ω ≠ C ω ∨ A ω = none} ≤ δL + δS + δU := by
  apply le_trans (measure_mono ?_) (three_failure_bound μ L S U δL δS δU hL hS hU)
  intro ω hw hg
  exact hw.elim (fun hn => hn (fields ω hg).1) (fields ω hg).2

theorem reports_to_scaled_differences (A C : ℕ → Ω → Option Report)
    (fallbackA fallbackC : ℕ → Ω → Report) (pointScale varianceScale : ℕ → ℝ)
    (good : ℕ → Set Ω)
    (fields : ∀ n ω, ω ∈ good n → A n ω = C n ω ∧ A n ω ≠ none)
    (hg : Tendsto (fun n => μ (good n)ᶜ) atTop (𝓝 0)) :
    TendstoInMeasure μ (fun n ω => pointScale n *
      ((extend (fallbackA n ω) (A n ω)).point - (extend (fallbackC n ω) (C n ω)).point)) atTop 0 ∧
    TendstoInMeasure μ (fun n ω => varianceScale n *
      ((extend (fallbackA n ω) (A n ω)).variance - (extend (fallbackC n ω) (C n ω)).variance)) atTop 0 ∧
    Tendsto (fun n => μ {ω | A n ω = none}) atTop (𝓝 0) := by
  have he : ∀ n ω, ω ∈ good n → extend (fallbackA n ω) (A n ω) =
      extend (fallbackC n ω) (C n ω) := by
    intro n ω hw
    obtain ⟨he, hn⟩ := fields n ω hw
    cases ha : A n ω with
    | none => exact (hn ha).elim
    | some r => rw [← he, ha]; rfl
  refine ⟨ErrorTarget.scaled_difference_of_high_probability_eq μ _ _ pointScale good
    (fun n ω hw => congrArg Report.point (he n ω hw)) hg,
    ErrorTarget.scaled_difference_of_high_probability_eq μ _ _ varianceScale good
    (fun n ω hw => congrArg Report.variance (he n ω hw)) hg, ?_⟩
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hg
  · intro n; exact bot_le
  · intro n
    apply measure_mono
    intro ω hn hw
    exact (fields n ω hw).2 hn

/-- Weaker entry: inspectable bounds on each of the TWO scaled errors on a good event. -/
theorem perturbation_transfer (pointA pointC varianceA varianceC : ℕ → Ω → ℝ)
    (pointScale varianceScale bPoint bVariance : ℕ → ℝ) (good : ℕ → Set Ω)
    (hp : ∀ n ω, ω ∈ good n → |pointScale n * (pointA n ω - pointC n ω)| ≤ bPoint n)
    (hv : ∀ n ω, ω ∈ good n → |varianceScale n * (varianceA n ω - varianceC n ω)| ≤ bVariance n)
    (hbp : Tendsto bPoint atTop (𝓝 0)) (hbv : Tendsto bVariance atTop (𝓝 0))
    (hg : Tendsto (fun n => μ (good n)ᶜ) atTop (𝓝 0)) :
    TendstoInMeasure μ (fun n ω => pointScale n * (pointA n ω - pointC n ω)) atTop 0 ∧
    TendstoInMeasure μ (fun n ω => varianceScale n * (varianceA n ω - varianceC n ω)) atTop 0 :=
  ⟨FixedCenterM.inMeasure_of_good_envelope μ _ bPoint good hp hbp hg,
    FixedCenterM.inMeasure_of_good_envelope μ _ bVariance good hv hbv hg⟩

end Probability

#print axioms Report
#print axioms members
#print axioms members_eq
#print axioms Pipeline
#print axioms finalSets
#print axioms learned
#print axioms reference
#print axioms actual
#print axioms learned_relabel
#print axioms learned_eq_reference
#print axioms actual_eq_reference
#print axioms actual_rejected
#print axioms labelEvent
#print axioms stableEvent
#print axioms usableEvent
#print axioms fields_to_report
#print axioms extend
#print axioms three_failure_bound
#print axioms three_failure_tendsto
#print axioms report_failure_bound
#print axioms reports_to_scaled_differences
#print axioms perturbation_transfer
end BinaryFormal.LearnedRuleApplication
