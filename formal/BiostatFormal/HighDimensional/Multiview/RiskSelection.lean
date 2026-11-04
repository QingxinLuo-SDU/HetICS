import BiostatFormal.HighDimensional.RiskSelection
import BiostatFormal.HighDimensional.Multiview.Guards
import BiostatFormal.HighDimensional.Multiview.Risks

/-!
# Bridge from abstract risk selection to multiview records

This file instantiates the already-proved deterministic risk-gap theorem on the subtype of records
in a realized guarded pool.  Candidate coverage, uniform-deviation probabilities, compatibility,
refinement, and B2 specialization remain hypotheses or later modules.
-/

namespace BiostatFormal.HighDimensional.Multiview

universe uCenter uLabel uSlot uContext

/-- Informal `η_sel = η_tie + η_guard + η_num` ↔ Lean: charge each tolerance exactly once. -/
def selectionTolerance (ηTie ηGuard ηNum : ℝ) : ℝ :=
  ηTie + ηGuard + ηNum

theorem selectionTolerance_nonneg
    {ηTie ηGuard ηNum : ℝ}
    (hTie : 0 ≤ ηTie) (hGuard : 0 ≤ ηGuard) (hNum : 0 ≤ ηNum) :
    0 ≤ selectionTolerance ηTie ηGuard ηNum := by
  unfold selectionTolerance
  positivity

/-- A contextual record is exact exactly when its label projection has zero source label distance. -/
theorem mem_exactCandidateSet_iff_labelDistance_eq_zero
    {Center : Type uCenter} {Label : Type uLabel}
    [Fintype Center] [DecidableEq Center] [Nonempty Center]
    [Fintype Label] [DecidableEq Label]
    {Slot : Type uSlot} {Context : Type uContext}
    {truth : Partition Center Label}
    {record : CandidateRecord Center Label Slot Context} :
    record ∈ exactCandidateSet truth ↔ labelDistance record.label truth = 0 := by
  change Exact record.label truth ↔ labelDistance record.label truth = 0
  exact exact_iff_labelDistance_eq_zero

/--
Instantiate `HighDimensional.exact_of_uniform_deviation_and_margin` on records in one realized
guarded pool.  Population and empirical risks use the same `CompositeRiskPair` and consume the
complete contextual record; exactness reads only its label projection.

The selected object has type `GuardedPoolMember pool`, not `SelectionDecision`, so `ABSTAIN` is
outside both the risk domain and the label-recovery conclusion.
-/
theorem exact_of_guarded_risk_selection
    {Center : Type uCenter} {Label : Type uLabel}
    [Fintype Center] [DecidableEq Center]
    [Fintype Label] [DecidableEq Label]
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (zGood zSelected : GuardedPoolMember pool)
    (u ηTie ηGuard ηNum : ℝ)
    (hExactWitness : zGood.1 ∈ exactCandidateSet truth)
    (hDeviation : ∀ z : GuardedPoolMember pool,
      |empiricalCompositeRisk risk z.1 - populationCompositeRisk risk z.1| ≤ u)
    (hMargin : ∀ z : GuardedPoolMember pool,
      z.1 ∈ badCandidateSet truth →
        populationCompositeRisk risk zGood.1 + 2 * u +
            selectionTolerance ηTie ηGuard ηNum <
          populationCompositeRisk risk z.1)
    (hSelected : empiricalCompositeRisk risk zSelected.1 ≤
      empiricalCompositeRisk risk zGood.1 + selectionTolerance ηTie ηGuard ηNum) :
    zSelected.1 ∈ exactCandidateSet truth := by
  apply BiostatFormal.HighDimensional.exact_of_uniform_deviation_and_margin
    (fun z : GuardedPoolMember pool => z.1 ∈ exactCandidateSet truth)
    (fun z : GuardedPoolMember pool => populationCompositeRisk risk z.1)
    (fun z : GuardedPoolMember pool => empiricalCompositeRisk risk z.1)
    zGood zSelected u (selectionTolerance ηTie ηGuard ηNum)
    hExactWitness hDeviation
  · intro z _ hNotExact
    exact hMargin z hNotExact
  · exact hSelected

/--
Source-facing form of `exact_of_guarded_risk_selection`: the selected guarded record has zero
permutation-invariant label distance.  This is a bridge corollary, not a B2 recovery theorem.
-/
theorem labelDistance_eq_zero_of_guarded_risk_selection
    {Center : Type uCenter} {Label : Type uLabel}
    [Fintype Center] [DecidableEq Center] [Nonempty Center]
    [Fintype Label] [DecidableEq Label]
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (zGood zSelected : GuardedPoolMember pool)
    (u ηTie ηGuard ηNum : ℝ)
    (hExactWitness : zGood.1 ∈ exactCandidateSet truth)
    (hDeviation : ∀ z : GuardedPoolMember pool,
      |empiricalCompositeRisk risk z.1 - populationCompositeRisk risk z.1| ≤ u)
    (hMargin : ∀ z : GuardedPoolMember pool,
      z.1 ∈ badCandidateSet truth →
        populationCompositeRisk risk zGood.1 + 2 * u +
            selectionTolerance ηTie ηGuard ηNum <
          populationCompositeRisk risk z.1)
    (hSelected : empiricalCompositeRisk risk zSelected.1 ≤
      empiricalCompositeRisk risk zGood.1 + selectionTolerance ηTie ηGuard ηNum) :
    labelDistance zSelected.1.label truth = 0 := by
  apply mem_exactCandidateSet_iff_labelDistance_eq_zero.mp
  exact exact_of_guarded_risk_selection pool risk truth zGood zSelected
    u ηTie ηGuard ηNum hExactWitness hDeviation hMargin hSelected

end BiostatFormal.HighDimensional.Multiview
