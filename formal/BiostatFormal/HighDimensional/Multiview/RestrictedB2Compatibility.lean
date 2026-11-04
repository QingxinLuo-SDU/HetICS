import BiostatFormal.HighDimensional.Multiview.RiskSelection
import Mathlib.Data.Finset.Max
import Mathlib.MeasureTheory.Measure.Real
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Restricted B2 compatibility

This file formalizes the deterministic interface isolated by the B2, `K = 3` compatibility audit.
It does not assert that the frozen B2 generator supplies the separation, heterogeneity, robust-
normalization, or oracle-to-record constants.  Those remain explicit upstream hypotheses.

The unrestricted, boundary-inclusive B2 compatibility claim is false.  In particular, none of the
declarations below may be read as reinstating it.
-/

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

universe uCenter uLabel uSlot uContext uSample

noncomputable section

/-- Exact records inside one realized finite guarded pool. -/
def exactGuardedRecordSet
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label) :
    Finset (CandidateRecord Center Label Slot Context) := by
  classical
  exact pool.filter fun record => record ∈ exactCandidateSet truth

/-- Bad records inside one realized finite guarded pool. -/
def badGuardedRecordSet
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label) :
    Finset (CandidateRecord Center Label Slot Context) := by
  classical
  exact pool.filter fun record => record ∈ badCandidateSet truth

@[simp]
theorem mem_exactGuardedRecordSet_iff
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    {pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context)}
    {truth : Partition Center Label}
    {record : CandidateRecord Center Label Slot Context} :
    record ∈ exactGuardedRecordSet pool truth ↔
      record ∈ pool ∧ record ∈ exactCandidateSet truth := by
  classical
  simp [exactGuardedRecordSet]

@[simp]
theorem mem_badGuardedRecordSet_iff
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    {pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context)}
    {truth : Partition Center Label}
    {record : CandidateRecord Center Label Slot Context} :
    record ∈ badGuardedRecordSet pool truth ↔
      record ∈ pool ∧ record ∈ badCandidateSet truth := by
  classical
  simp [badGuardedRecordSet]

/-- All records in the comparison pool carry the same frozen `RiskContext` value. -/
def guardedPoolHasRiskContext
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (context : Context) : Prop :=
  ∀ record ∈ pool, record.context = context

/--
The supported restricted B2 domain: three labels, the balanced `E = 9` or `E = 12` roster, and one
shared frozen risk context.  Minimum-cluster and view eligibility are represented upstream by
membership in the realized guarded pool and by the impurity lower-bound hypothesis.
-/
structure RestrictedB2Domain
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [Fintype Center] [Fintype Label]
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (context : Context) : Prop where
  threeLabels : Fintype.card Label = 3
  supportedCenterCount : Fintype.card Center = 9 ∨ Fintype.card Center = 12
  sameRiskContext : guardedPoolHasRiskContext pool context

/-- The exact finite-pool reference risk: the minimum population composite risk over exact records. -/
def exactReferenceRisk
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (hExact : (exactGuardedRecordSet pool truth).Nonempty) : ℝ :=
  ((exactGuardedRecordSet pool truth).image (populationCompositeRisk risk)).min'
    (hExact.image (populationCompositeRisk risk))

/-- The realized population gap `Delta_MV`: the minimum bad-record excess over the exact reference. -/
def recordRiskGap
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (hExact : (exactGuardedRecordSet pool truth).Nonempty)
    (hBad : (badGuardedRecordSet pool truth).Nonempty) : ℝ :=
  ((badGuardedRecordSet pool truth).image fun record =>
      populationCompositeRisk risk record - exactReferenceRisk pool risk truth hExact).min'
    (hBad.image fun record =>
      populationCompositeRisk risk record - exactReferenceRisk pool risk truth hExact)

theorem exists_exact_record_populationRisk_eq_exactReferenceRisk
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (hExact : (exactGuardedRecordSet pool truth).Nonempty) :
    ∃ record ∈ exactGuardedRecordSet pool truth,
      populationCompositeRisk risk record = exactReferenceRisk pool risk truth hExact := by
  have hMember := Finset.min'_mem
    ((exactGuardedRecordSet pool truth).image (populationCompositeRisk risk))
    (hExact.image (populationCompositeRisk risk))
  rcases Finset.mem_image.mp hMember with ⟨record, hRecord, hRisk⟩
  exact ⟨record, hRecord, hRisk⟩

theorem exactReferenceRisk_le_populationRisk
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (hExact : (exactGuardedRecordSet pool truth).Nonempty)
    {record : CandidateRecord Center Label Slot Context}
    (hRecord : record ∈ exactGuardedRecordSet pool truth) :
    exactReferenceRisk pool risk truth hExact ≤ populationCompositeRisk risk record := by
  unfold exactReferenceRisk
  apply Finset.min'_le
  exact Finset.mem_image.mpr ⟨record, hRecord, rfl⟩

theorem lower_le_recordRiskGap
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (hExact : (exactGuardedRecordSet pool truth).Nonempty)
    (hBad : (badGuardedRecordSet pool truth).Nonempty)
    (lower : ℝ)
    (hLower : ∀ record ∈ badGuardedRecordSet pool truth,
      lower ≤ populationCompositeRisk risk record - exactReferenceRisk pool risk truth hExact) :
    lower ≤ recordRiskGap pool risk truth hExact hBad := by
  unfold recordRiskGap
  apply Finset.le_min'
  intro value hValue
  rcases Finset.mem_image.mp hValue with ⟨record, hRecord, rfl⟩
  exact hLower record hRecord

/-- Audited K=3 confusion-matrix constants for the two frozen center counts. -/
def restrictedB2Chi (centerCount : ℕ) : ℝ :=
  if centerCount = 9 then 1 / 2
  else if centerCount = 12 then 1 / 3
  else 0

@[simp]
theorem restrictedB2Chi_nine : restrictedB2Chi 9 = (1 / 2 : ℝ) := by
  norm_num [restrictedB2Chi]

@[simp]
theorem restrictedB2Chi_twelve : restrictedB2Chi 12 = (1 / 3 : ℝ) := by
  norm_num [restrictedB2Chi]

theorem restrictedB2Chi_pos
    {centerCount : ℕ} (hCenterCount : centerCount = 9 ∨ centerCount = 12) :
    0 < restrictedB2Chi centerCount := by
  rcases hCenterCount with rfl | rfl <;> norm_num

/-- `kappa_MV = E chi(E) g0` on the restricted B2 domain. -/
def restrictedB2CompatibilitySlope (centerCount : ℕ) (g₀ : ℝ) : ℝ :=
  (centerCount : ℝ) * restrictedB2Chi centerCount * g₀

/-- Total compatibility defect `B_het + 2 b0`. -/
def restrictedB2CompatibilityBias (BHet b₀ : ℝ) : ℝ :=
  BHet + 2 * b₀

theorem restrictedB2CompatibilitySlope_pos
    {Center : Type uCenter} [Fintype Center]
    (hCenterCount : Fintype.card Center = 9 ∨ Fintype.card Center = 12)
    {g₀ : ℝ} (hg₀ : 0 < g₀) :
    0 < restrictedB2CompatibilitySlope (Fintype.card Center) g₀ := by
  unfold restrictedB2CompatibilitySlope
  have hCardPos : (0 : ℝ) < Fintype.card Center := by
    rcases hCenterCount with hNine | hTwelve
    · rw [hNine]
      norm_num
    · rw [hTwelve]
      norm_num
  exact mul_pos (mul_pos hCardPos (restrictedB2Chi_pos hCenterCount)) hg₀

/-- The record-level compatibility certificate on one finite guarded pool and frozen context. -/
structure RestrictedB2Compatibility
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [Fintype Center] [DecidableEq Center]
    [Fintype Label] [DecidableEq Label]
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (context : Context)
    (hExact : (exactGuardedRecordSet pool truth).Nonempty)
    (g₀ BHet b₀ : ℝ) : Prop where
  g₀_pos : 0 < g₀
  heterogeneityBudget_nonneg : 0 ≤ BHet
  oracleApproximationBudget_nonneg : 0 ≤ b₀
  sameRiskContext : guardedPoolHasRiskContext pool context
  bound : ∀ record ∈ badGuardedRecordSet pool truth,
      restrictedB2CompatibilitySlope (Fintype.card Center) g₀ *
            labelDistance record.label truth -
          restrictedB2CompatibilityBias BHet b₀ ≤
        populationCompositeRisk risk record - exactReferenceRisk pool risk truth hExact

/--
The restricted G-06 algebra.  `hStructural` is the audited oracle/normalization/record bridge
`risk excess >= gMin * impurity - (BHet + 2*b0)`.  `hImpuritySlope` is the K=3 confusion-matrix
bound with `chi(9)=1/2` or `chi(12)=1/3`.  Neither premise is claimed here for the B2 generator.
-/
theorem restrictedB2Compatibility_of_structural_bounds
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [Fintype Center] [DecidableEq Center]
    [Fintype Label] [DecidableEq Label]
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (context : Context)
    (domain : RestrictedB2Domain pool context)
    (hExact : (exactGuardedRecordSet pool truth).Nonempty)
    (impurity : CandidateRecord Center Label Slot Context → ℝ)
    (gMin g₀ BHet b₀ : ℝ)
    (hg₀ : 0 < g₀)
    (hgMin : g₀ ≤ gMin)
    (hBHet : 0 ≤ BHet)
    (hb₀ : 0 ≤ b₀)
    (hImpurityNonneg : ∀ record ∈ badGuardedRecordSet pool truth, 0 ≤ impurity record)
    (hImpuritySlope : ∀ record ∈ badGuardedRecordSet pool truth,
      (Fintype.card Center : ℝ) * restrictedB2Chi (Fintype.card Center) *
          labelDistance record.label truth ≤ impurity record)
    (hStructural : ∀ record ∈ badGuardedRecordSet pool truth,
      gMin * impurity record - restrictedB2CompatibilityBias BHet b₀ ≤
        populationCompositeRisk risk record - exactReferenceRisk pool risk truth hExact) :
    RestrictedB2Compatibility pool risk truth context hExact g₀ BHet b₀ := by
  refine ⟨hg₀, hBHet, hb₀, domain.sameRiskContext, ?_⟩
  intro record hRecord
  calc
    restrictedB2CompatibilitySlope (Fintype.card Center) g₀ *
          labelDistance record.label truth - restrictedB2CompatibilityBias BHet b₀ =
        g₀ * ((Fintype.card Center : ℝ) * restrictedB2Chi (Fintype.card Center) *
          labelDistance record.label truth) - restrictedB2CompatibilityBias BHet b₀ := by
            unfold restrictedB2CompatibilitySlope
            ring
    _ ≤ g₀ * impurity record - restrictedB2CompatibilityBias BHet b₀ := by
      exact sub_le_sub_right
        (mul_le_mul_of_nonneg_left (hImpuritySlope record hRecord) hg₀.le) _
    _ ≤ gMin * impurity record - restrictedB2CompatibilityBias BHet b₀ := by
      exact sub_le_sub_right
        (mul_le_mul_of_nonneg_right hgMin (hImpurityNonneg record hRecord)) _
    _ ≤ populationCompositeRisk risk record - exactReferenceRisk pool risk truth hExact :=
      hStructural record hRecord

/-- Every bad partition has at least one center error after the best global label permutation. -/
theorem one_div_card_le_labelDistance_of_bad
    {Center : Type uCenter} {Label : Type uLabel}
    [Fintype Center] [DecidableEq Center] [Nonempty Center]
    [Fintype Label] [DecidableEq Label]
    {candidate truth : Partition Center Label}
    (hBad : ¬Exact candidate truth) :
    1 / (Fintype.card Center : ℝ) ≤ labelDistance candidate truth := by
  have hMismatchNe : optimalMismatchCount candidate truth ≠ 0 := by
    intro hZero
    exact hBad (labelEquivalent_iff_optimalMismatchCount_eq_zero.mpr hZero)
  have hOneNat : 1 ≤ optimalMismatchCount candidate truth :=
    Nat.one_le_iff_ne_zero.mpr hMismatchNe
  have hOneReal : (1 : ℝ) ≤ optimalMismatchCount candidate truth := by
    exact_mod_cast hOneNat
  have hCardPos : (0 : ℝ) < Fintype.card Center := by
    exact_mod_cast Fintype.card_pos
  simpa [labelDistance] using div_le_div_of_nonneg_right hOneReal hCardPos.le

/-- `Delta_MV >= kappa_MV/E - (BHet + 2*b0)` for a nonempty finite bad-record set. -/
theorem restrictedB2_recordRiskGap_lower_bound
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [Fintype Center] [DecidableEq Center] [Nonempty Center]
    [Fintype Label] [DecidableEq Label]
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (context : Context)
    (domain : RestrictedB2Domain pool context)
    (hExact : (exactGuardedRecordSet pool truth).Nonempty)
    (hBad : (badGuardedRecordSet pool truth).Nonempty)
    (g₀ BHet b₀ : ℝ)
    (hCompatibility : RestrictedB2Compatibility pool risk truth context hExact g₀ BHet b₀) :
    restrictedB2CompatibilitySlope (Fintype.card Center) g₀ /
          (Fintype.card Center : ℝ) - restrictedB2CompatibilityBias BHet b₀ ≤
      recordRiskGap pool risk truth hExact hBad := by
  apply lower_le_recordRiskGap pool risk truth hExact hBad
  intro record hRecord
  have hBadRecord : record ∈ badCandidateSet truth :=
    (mem_badGuardedRecordSet_iff.mp hRecord).2
  have hDistance := one_div_card_le_labelDistance_of_bad
    (candidate := record.label) (truth := truth) hBadRecord
  have hSlopeNonneg :
      0 ≤ restrictedB2CompatibilitySlope (Fintype.card Center) g₀ :=
    (restrictedB2CompatibilitySlope_pos domain.supportedCenterCount hCompatibility.g₀_pos).le
  calc
    restrictedB2CompatibilitySlope (Fintype.card Center) g₀ /
          (Fintype.card Center : ℝ) - restrictedB2CompatibilityBias BHet b₀ =
        restrictedB2CompatibilitySlope (Fintype.card Center) g₀ *
          (1 / (Fintype.card Center : ℝ)) - restrictedB2CompatibilityBias BHet b₀ := by
            ring
    _ ≤ restrictedB2CompatibilitySlope (Fintype.card Center) g₀ *
          labelDistance record.label truth - restrictedB2CompatibilityBias BHet b₀ := by
      exact sub_le_sub_right (mul_le_mul_of_nonneg_left hDistance hSlopeNonneg) _
    _ ≤ populationCompositeRisk risk record - exactReferenceRisk pool risk truth hExact :=
      hCompatibility.bound record hRecord

/-- Source-facing record-level strict population-gap predicate, the deterministic content of `G`. -/
def StrictPopulationGap
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (u η : ℝ) : Prop :=
  ∃ goodRecord ∈ exactGuardedRecordSet pool truth,
    ∀ badRecord ∈ badGuardedRecordSet pool truth,
      populationCompositeRisk risk goodRecord + 2 * u + η <
        populationCompositeRisk risk badRecord

/-- The selector threshold turns restricted compatibility into the strict population-gap event. -/
theorem strictPopulationGap_of_restrictedB2Compatibility
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [Fintype Center] [DecidableEq Center] [Nonempty Center]
    [Fintype Label] [DecidableEq Label]
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (context : Context)
    (domain : RestrictedB2Domain pool context)
    (hExact : (exactGuardedRecordSet pool truth).Nonempty)
    (g₀ BHet b₀ u η : ℝ)
    (hCompatibility : RestrictedB2Compatibility pool risk truth context hExact g₀ BHet b₀)
    (hSelectorMargin :
      2 * u + η < restrictedB2CompatibilitySlope (Fintype.card Center) g₀ /
        (Fintype.card Center : ℝ) - restrictedB2CompatibilityBias BHet b₀) :
    StrictPopulationGap pool risk truth u η := by
  rcases exists_exact_record_populationRisk_eq_exactReferenceRisk pool risk truth hExact with
    ⟨goodRecord, hGoodRecord, hGoodRisk⟩
  refine ⟨goodRecord, hGoodRecord, ?_⟩
  intro badRecord hBadRecord
  have hBadCandidate : badRecord ∈ badCandidateSet truth :=
    (mem_badGuardedRecordSet_iff.mp hBadRecord).2
  have hDistance := one_div_card_le_labelDistance_of_bad
    (candidate := badRecord.label) (truth := truth) hBadCandidate
  have hSlopeNonneg :
      0 ≤ restrictedB2CompatibilitySlope (Fintype.card Center) g₀ :=
    (restrictedB2CompatibilitySlope_pos domain.supportedCenterCount hCompatibility.g₀_pos).le
  have hRiskLower := hCompatibility.bound badRecord hBadRecord
  have hAtLeastOneError :
      restrictedB2CompatibilitySlope (Fintype.card Center) g₀ /
            (Fintype.card Center : ℝ) - restrictedB2CompatibilityBias BHet b₀ ≤
        populationCompositeRisk risk badRecord - exactReferenceRisk pool risk truth hExact := by
    calc
      restrictedB2CompatibilitySlope (Fintype.card Center) g₀ /
            (Fintype.card Center : ℝ) - restrictedB2CompatibilityBias BHet b₀ =
          restrictedB2CompatibilitySlope (Fintype.card Center) g₀ *
            (1 / (Fintype.card Center : ℝ)) - restrictedB2CompatibilityBias BHet b₀ := by
              ring
      _ ≤ restrictedB2CompatibilitySlope (Fintype.card Center) g₀ *
            labelDistance badRecord.label truth - restrictedB2CompatibilityBias BHet b₀ := by
        exact sub_le_sub_right (mul_le_mul_of_nonneg_left hDistance hSlopeNonneg) _
      _ ≤ populationCompositeRisk risk badRecord - exactReferenceRisk pool risk truth hExact :=
        hRiskLower
  rw [hGoodRisk]
  linarith

/--
The strict-gap predicate is exactly the missing input needed by the already-verified guarded risk
selector.  This theorem reuses that selector; it does not reprove its deviation argument.
-/
theorem exact_of_strictPopulationGap
    {Center : Type uCenter} {Label : Type uLabel}
    [Fintype Center] [DecidableEq Center]
    [Fintype Label] [DecidableEq Label]
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (zSelected : GuardedPoolMember pool)
    (u ηTie ηGuard ηNum : ℝ)
    (hGap : StrictPopulationGap pool risk truth u
      (selectionTolerance ηTie ηGuard ηNum))
    (hDeviation : ∀ record : GuardedPoolMember pool,
      |empiricalCompositeRisk risk record.1 - populationCompositeRisk risk record.1| ≤ u)
    (hSelected : ∀ record : GuardedPoolMember pool,
      empiricalCompositeRisk risk zSelected.1 ≤
        empiricalCompositeRisk risk record.1 + selectionTolerance ηTie ηGuard ηNum) :
    zSelected.1 ∈ exactCandidateSet truth := by
  rcases hGap with ⟨goodRecord, hGoodRecord, hStrict⟩
  have hGoodPool : goodRecord ∈ pool :=
    (mem_exactGuardedRecordSet_iff.mp hGoodRecord).1
  have hGoodExact : goodRecord ∈ exactCandidateSet truth :=
    (mem_exactGuardedRecordSet_iff.mp hGoodRecord).2
  let zGood : GuardedPoolMember pool := ⟨goodRecord, hGoodPool⟩
  apply exact_of_guarded_risk_selection pool risk truth zGood zSelected
    u ηTie ηGuard ηNum hGoodExact hDeviation
  · intro badRecord hBadRecord
    exact hStrict badRecord.1
      (mem_badGuardedRecordSet_iff.mpr ⟨badRecord.2, hBadRecord⟩)
  · exact hSelected zGood

/-- Pointwise strict-gap event for a random realized pool and frozen contextual risk. -/
def strictPopulationGapEvent
    {Sample : Type uSample}
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : Sample → GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : Sample → CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (u η : Sample → ℝ) : Set Sample :=
  {sample | StrictPopulationGap (pool sample) (risk sample) truth (u sample) (η sample)}

/--
The total source event `G`: if no label-returning record exists, the event succeeds off the return
event; on the return event it requires the strict population gap.
-/
def restrictedB2GapEvent
    {Sample : Type uSample}
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (returnEvent : Set Sample)
    (pool : Sample → GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : Sample → CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (u η : Sample → ℝ) : Set Sample :=
  returnEventᶜ ∪ strictPopulationGapEvent pool risk truth u η

/-- `delta_G` is an upper bound on failure of the total compatibility/gap event, not a DGP constant. -/
def ControlsRestrictedB2GapFailure
    {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) (G : Set Sample) (δG : ENNReal) : Prop :=
  μ Gᶜ ≤ δG

/--
If a primitive restricted-parameter event implies the strict gap whenever a label is returned,
its failure budget also controls failure of `G`.  A numerical or analytic bound on the primitive
event remains an upstream B2 obligation.
-/
theorem controlsRestrictedB2GapFailure_of_assumptionEvent
    {Sample : Type uSample} [MeasurableSpace Sample]
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (μ : Measure Sample)
    (returnEvent assumptionEvent : Set Sample)
    (pool : Sample → GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (risk : Sample → CompositeRiskPair (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (u η : Sample → ℝ)
    (δG : ENNReal)
    (hGapOnAssumptions : ∀ sample ∈ assumptionEvent,
      sample ∈ returnEvent →
        StrictPopulationGap (pool sample) (risk sample) truth (u sample) (η sample))
    (hAssumptionFailure : μ assumptionEventᶜ ≤ δG) :
    ControlsRestrictedB2GapFailure μ
      (restrictedB2GapEvent returnEvent pool risk truth u η) δG := by
  have hAssumptionsSubset :
      assumptionEvent ⊆ restrictedB2GapEvent returnEvent pool risk truth u η := by
    intro sample hSample
    by_cases hReturn : sample ∈ returnEvent
    · exact Set.mem_union_right _ (hGapOnAssumptions sample hSample hReturn)
    · exact Set.mem_union_left _ hReturn
  apply (measure_mono ?_).trans hAssumptionFailure
  intro sample hGapFailure hAssumptions
  exact hGapFailure (hAssumptionsSubset hAssumptions)

end

end BiostatFormal.HighDimensional.Multiview
