import BiostatFormal.HighDimensional.Multiview.B2V3ActualRefinement

/-!
# Actual B2-v3 A3 primitive tails

The fixed witness subledger has `2(T+1)+1` sites: `T` ordinary calls and one candidate-local
terminalization in each direction, followed by one paired atomic guard.  Seven source-facing
failure atoms are grouped into the four published A3 budgets.  First-failure events are conditioned
on the realized prefix, so no independence between adaptive calls is assumed.
-/

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

universe uOmega

/-- The seven low-level failures traced in the actual candidate/guard source path. -/
inductive B2V3ActualRawFailureKind where
  | score
  | rsc
  | fitResidual
  | margin
  | regret
  | candidateTerminal
  | guard
  deriving DecidableEq, Fintype, Repr

namespace B2V3ActualRawFailureKind

/-- Proven grouping into the existing four public A3 budget categories. -/
def publicKind : B2V3ActualRawFailureKind → B2V3PrimitiveFailureKind
  | .score | .rsc => .scoreRSC
  | .fitResidual => .fit
  | .margin | .regret => .assignment
  | .candidateTerminal | .guard => .terminal

/-- Source-compatible atom/site pairs for the fixed witness subledger. -/
def compatible {T : Nat} : B2V3ActualRawFailureKind → B2V3ActualWitnessSite T → Prop
  | .score, .ordinary _ _
  | .rsc, .ordinary _ _
  | .fitResidual, .ordinary _ _
  | .margin, .ordinary _ _
  | .regret, .ordinary _ _
  | .candidateTerminal, .terminal _
  | .guard, .guard => True
  | _, _ => False

end B2V3ActualRawFailureKind

/-- Number of sites in the preregistered `outcome__kmeans` witness envelope. -/
def b2V3ActualWitnessLedgerSize (T : Nat) : Nat := 2 * (T + 1) + 1

/--
An ordered fixed ledger for the actual witness.  `siteOrder` binds every numerical ledger index to
one source site.  Canonical deduplication can deactivate a site but cannot add indices.
-/
structure B2V3ActualPrimitiveLedger (Omega : Type uOmega) (T : Nat) where
  siteOrder : Fin (b2V3ActualWitnessLedgerSize T) ≃ B2V3ActualWitnessSite T
  active : Fin (b2V3ActualWitnessLedgerSize T) → Set Omega
  failure : B2V3ActualRawFailureKind →
    Fin (b2V3ActualWitnessLedgerSize T) → Set Omega
  failure_subset_active : ∀ kind q, failure kind q ⊆ active q
  incompatible_empty : ∀ kind q,
    ¬ kind.compatible (siteOrder q) → failure kind q = ∅

namespace B2V3ActualPrimitiveLedger

/-- The ledger index type has exactly the source-derived witness-site cardinality. -/
theorem fixed_site_count (T : Nat) :
    Fintype.card (Fin (b2V3ActualWitnessLedgerSize T)) = 2 * (T + 1) + 1 := by
  exact Fintype.card_fin _

/-- All raw primitive atoms at earlier realized sites are good. -/
def prefixGood {Omega : Type uOmega} {T : Nat}
    (ledger : B2V3ActualPrimitiveLedger Omega T)
    (q : Fin (b2V3ActualWitnessLedgerSize T)) : Set Omega :=
  {omega | ∀ p, p.val < q.val → ∀ kind, omega ∉ ledger.failure kind p}

/-- A raw failure at the first bad active source site, inside the attraction event. -/
def firstFailure {Omega : Type uOmega} {T : Nat}
    (CAttr : Set Omega) (ledger : B2V3ActualPrimitiveLedger Omega T)
    (kind : B2V3ActualRawFailureKind)
    (q : Fin (b2V3ActualWitnessLedgerSize T)) : Set Omega :=
  CAttr ∩ ledger.prefixGood q ∩ ledger.active q ∩ ledger.failure kind q

/-- Union of all seven first-failure atoms over the fixed witness ledger. -/
def allFirstFailures {Omega : Type uOmega} {T : Nat}
    (CAttr : Set Omega) (ledger : B2V3ActualPrimitiveLedger Omega T) : Set Omega :=
  ⋃ q, ⋃ kind, ledger.firstFailure CAttr kind q

/--
If raw-good source bindings yield `Rsrc`, then every attraction-and-source failure has a first raw
failure site.  This lemma is purely finite and uses no independence.
-/
theorem sourceFailure_subset_allFirstFailures
    {Omega : Type uOmega} {T : Nat}
    (CAttr Rsrc : Set Omega) (ledger : B2V3ActualPrimitiveLedger Omega T)
    (hRawGoodSurvives : ∀ omega, omega ∈ CAttr →
      (∀ q kind, omega ∉ ledger.failure kind q) → omega ∈ Rsrc) :
    CAttr ∩ Rsrcᶜ ⊆ ledger.allFirstFailures CAttr := by
  classical
  intro omega hFailure
  have hAttr : omega ∈ CAttr := hFailure.1
  have hNotSurvived : omega ∉ Rsrc := hFailure.2
  have hSomeFailure : ∃ q, ∃ kind, omega ∈ ledger.failure kind q := by
    by_contra hNone
    apply hNotSurvived
    apply hRawGoodSurvives omega hAttr
    intro q kind hAt
    exact hNone ⟨q, kind, hAt⟩
  let failingCalls : Finset (Fin (b2V3ActualWitnessLedgerSize T)) :=
    Finset.univ.filter fun q => ∃ kind, omega ∈ ledger.failure kind q
  have hFailingCallsNonempty : failingCalls.Nonempty := by
    rcases hSomeFailure with ⟨q, kind, hAt⟩
    exact ⟨q, Finset.mem_filter.mpr ⟨Finset.mem_univ q, ⟨kind, hAt⟩⟩⟩
  let firstCall := failingCalls.min' hFailingCallsNonempty
  have hFirstCallMem : firstCall ∈ failingCalls := by
    simpa [firstCall] using Finset.min'_mem failingCalls hFailingCallsNonempty
  rcases (Finset.mem_filter.mp hFirstCallMem).2 with ⟨firstKind, hFirstFailure⟩
  have hPrefix : omega ∈ ledger.prefixGood firstCall := by
    intro earlier hEarlier kind hEarlierFailure
    have hEarlierMem : earlier ∈ failingCalls :=
      Finset.mem_filter.mpr
        ⟨Finset.mem_univ earlier, ⟨kind, hEarlierFailure⟩⟩
    have hMin : firstCall ≤ earlier := by
      simpa [firstCall] using Finset.min'_le failingCalls earlier hEarlierMem
    exact (not_le_of_gt hEarlier) hMin
  have hActive : omega ∈ ledger.active firstCall :=
    ledger.failure_subset_active firstKind firstCall hFirstFailure
  rw [allFirstFailures]
  refine Set.mem_iUnion.2 ⟨firstCall, Set.mem_iUnion.2 ⟨firstKind, ?_⟩⟩
  exact ⟨⟨⟨hAttr, hPrefix⟩, hActive⟩, hFirstFailure⟩

end B2V3ActualPrimitiveLedger

/-- Conditional-on-prefix raw tail bounds for all seven source atoms. -/
structure B2V3ActualRawTailBudgets
    {Omega : Type uOmega} [MeasurableSpace Omega]
    (mu : Measure Omega) {T : Nat}
    (CAttr : Set Omega) (ledger : B2V3ActualPrimitiveLedger Omega T) where
  delta : B2V3ActualRawFailureKind →
    Fin (b2V3ActualWitnessLedgerSize T) → ENNReal
  firstFailure_le : ∀ kind q,
    mu (ledger.firstFailure CAttr kind q) ≤ delta kind q

namespace B2V3ActualRawTailBudgets

variable {Omega : Type uOmega} [MeasurableSpace Omega]
  {mu : Measure Omega} {T : Nat} {CAttr : Set Omega}
  {ledger : B2V3ActualPrimitiveLedger Omega T}

/-- The public per-site four-way grouping of the seven raw budgets. -/
noncomputable def groupedDelta
    (budgets : B2V3ActualRawTailBudgets mu CAttr ledger)
    (kind : B2V3PrimitiveFailureKind)
    (q : Fin (b2V3ActualWitnessLedgerSize T)) : ENNReal :=
  match kind with
  | .scoreRSC => budgets.delta .score q + budgets.delta .rsc q
  | .fit => budgets.delta .fitResidual q
  | .assignment => budgets.delta .margin q + budgets.delta .regret q
  | .terminal => budgets.delta .candidateTerminal q + budgets.delta .guard q

/-- The actual public A3 budget, with the atomic guard included in `terminal`. -/
noncomputable def deltaR
    (budgets : B2V3ActualRawTailBudgets mu CAttr ledger) : ENNReal :=
  ∑ q, ∑ kind : B2V3PrimitiveFailureKind, budgets.groupedDelta kind q

/-- The published four-way budget is exactly the sum of the seven raw source tails. -/
theorem deltaR_eq_rawSum
    (budgets : B2V3ActualRawTailBudgets mu CAttr ledger) :
    budgets.deltaR = ∑ q, ∑ kind : B2V3ActualRawFailureKind, budgets.delta kind q := by
  rw [deltaR]
  apply Finset.sum_congr rfl
  intro q _
  rw [show (Finset.univ : Finset B2V3PrimitiveFailureKind) =
      {.scoreRSC, .fit, .assignment, .terminal} by decide]
  rw [show (Finset.univ : Finset B2V3ActualRawFailureKind) =
      {.score, .rsc, .fitResidual, .margin, .regret, .candidateTerminal, .guard} by decide]
  simp [groupedDelta]
  ac_rfl

end B2V3ActualRawTailBudgets

/--
Actual fixed-ledger A3 probability interface.  Its only probability inputs are the seven raw
conditional tails; the result is reported through the four grouped public budgets.
-/
theorem measure_b2V3ActualSourceFailure_le_deltaR
    {Omega : Type uOmega} [MeasurableSpace Omega]
    (mu : Measure Omega) {T : Nat}
    (CAttr Rsrc : Set Omega)
    (ledger : B2V3ActualPrimitiveLedger Omega T)
    (budgets : B2V3ActualRawTailBudgets mu CAttr ledger)
    (hRawGoodSurvives : ∀ omega, omega ∈ CAttr →
      (∀ q kind, omega ∉ ledger.failure kind q) → omega ∈ Rsrc) :
    mu (CAttr ∩ Rsrcᶜ) ≤ budgets.deltaR := by
  calc
    mu (CAttr ∩ Rsrcᶜ) ≤ mu (ledger.allFirstFailures CAttr) :=
      measure_mono (ledger.sourceFailure_subset_allFirstFailures CAttr Rsrc hRawGoodSurvives)
    _ ≤ ∑ q, mu (⋃ kind, ledger.firstFailure CAttr kind q) := by
      unfold B2V3ActualPrimitiveLedger.allFirstFailures
      exact measure_iUnion_fintype_le mu _
    _ ≤ ∑ q, ∑ kind, mu (ledger.firstFailure CAttr kind q) := by
      apply Finset.sum_le_sum
      intro q _
      exact measure_iUnion_fintype_le mu _
    _ ≤ ∑ q, ∑ kind, budgets.delta kind q := by
      exact Finset.sum_le_sum fun q _ =>
        Finset.sum_le_sum fun kind _ => budgets.firstFailure_le kind q
    _ = budgets.deltaR := budgets.deltaR_eq_rawSum.symm

end BiostatFormal.HighDimensional.Multiview
