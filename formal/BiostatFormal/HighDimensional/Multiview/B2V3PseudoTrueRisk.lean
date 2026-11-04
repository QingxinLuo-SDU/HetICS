import BiostatFormal.HighDimensional.Multiview.B2V3ConditionalRisk
import BiostatFormal.HighDimensional.Multiview.RestrictedB2Compatibility
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# B2-v3 pseudo-true X/Y loss and restricted compatibility

This file makes the analytic input finer than an assumed record-risk gap.  Each active record has a
nonnegative directed off-diagonal class-pair mixing ledger.  X and Y have separate pseudo-true
pair-loss tables; their equal-weight fusion is calculated by finite-sum algebra.  A uniform fusion
pair gap then controls total mixing impurity.

View-specific heterogeneity and oracle-to-fitted-record errors remain explicit defects.  They are
absorbed into a bias-free compatibility slope only when the fusion pair gap dominates the defect on
the strictly positive bad-partition impurity floor.
-/

namespace BiostatFormal.HighDimensional.Multiview

noncomputable section

universe uCandidate uCenter uLabel uSlot uContext

/-- Directed off-diagonal true-class pairs.  A source ledger may split an unordered coefficient
between its two orientations; its convention must be used consistently in the risk decomposition. -/
def b2V3OffDiagonalLabelPairs
    (Label : Type uLabel) [Fintype Label] [DecidableEq Label] : Finset (Label × Label) :=
  Finset.univ.filter fun pair ↦ pair.1 ≠ pair.2

/-- Total cross-true-class mixing recorded for one candidate. -/
def b2V3TotalMixingMass
    {Candidate : Type uCandidate} {Label : Type uLabel}
    [Fintype Label] [DecidableEq Label]
    (mixingWeight : Candidate → Label × Label → ℝ) (candidate : Candidate) : ℝ :=
  ∑ pair ∈ b2V3OffDiagonalLabelPairs Label, mixingWeight candidate pair

/-- Pseudo-true mismatch loss in one view. -/
def b2V3PseudoTrueMismatchLoss
    {Candidate : Type uCandidate} {Label : Type uLabel}
    [Fintype Label] [DecidableEq Label]
    (mixingWeight : Candidate → Label × Label → ℝ)
    (pairLoss : Label × Label → ℝ) (candidate : Candidate) : ℝ :=
  ∑ pair ∈ b2V3OffDiagonalLabelPairs Label,
    mixingWeight candidate pair * pairLoss pair

/-- Exact one-view fit defect: the positive part of pseudo-true loss minus fitted risk excess. -/
def b2V3ViewFitDefect (pseudoTrueLoss fittedRiskExcess : ℝ) : ℝ :=
  max 0 (pseudoTrueLoss - fittedRiskExcess)

theorem b2V3ViewFitDefect_nonneg (pseudoTrueLoss fittedRiskExcess : ℝ) :
    0 ≤ b2V3ViewFitDefect pseudoTrueLoss fittedRiskExcess := by
  exact le_max_left _ _

/-- Subtracting the exact fit defect always bridges pseudo-true loss to fitted risk excess. -/
theorem b2V3PseudoTrueLoss_sub_viewFitDefect_le
    (pseudoTrueLoss fittedRiskExcess : ℝ) :
    pseudoTrueLoss - b2V3ViewFitDefect pseudoTrueLoss fittedRiskExcess ≤
      fittedRiskExcess := by
  have h : pseudoTrueLoss - fittedRiskExcess ≤
      b2V3ViewFitDefect pseudoTrueLoss fittedRiskExcess := by
    exact le_max_right _ _
  linarith

/-- A registered uniform defect bound supplies the bridge used by restricted compatibility. -/
theorem b2V3PseudoTrueLoss_sub_defectBudget_le
    (pseudoTrueLoss fittedRiskExcess defectBudget : ℝ)
    (hBudget : b2V3ViewFitDefect pseudoTrueLoss fittedRiskExcess ≤ defectBudget) :
    pseudoTrueLoss - defectBudget ≤ fittedRiskExcess := by
  linarith [b2V3PseudoTrueLoss_sub_viewFitDefect_le pseudoTrueLoss fittedRiskExcess]

/-- Equal-weight Y/X pair loss in the frozen common coordinate. -/
def b2V3FusionPairLoss
    {Label : Type uLabel}
    (outcomePairLoss treatmentPairLoss : Label × Label → ℝ)
    (pair : Label × Label) : ℝ :=
  (outcomePairLoss pair + treatmentPairLoss pair) / 2

/-- The registered restricted pair gap is the attained minimum over off-diagonal class pairs. -/
noncomputable def b2V3RestrictedGMin
    {Label : Type uLabel} [Fintype Label] [DecidableEq Label]
    (outcomePairLoss treatmentPairLoss : Label × Label → ℝ)
    (hPairs : (b2V3OffDiagonalLabelPairs Label).Nonempty) : ℝ :=
  ((b2V3OffDiagonalLabelPairs Label).image
    (b2V3FusionPairLoss outcomePairLoss treatmentPairLoss)).min'
      (hPairs.image _)

/-- Separate view lower bounds average once in the fusion pair gap. -/
theorem b2V3RestrictedGMin_ge_of_viewGaps
    {Label : Type uLabel} [Fintype Label] [DecidableEq Label]
    (outcomePairLoss treatmentPairLoss : Label × Label → ℝ)
    (hPairs : (b2V3OffDiagonalLabelPairs Label).Nonempty)
    (gY gX : ℝ)
    (hY : ∀ pair ∈ b2V3OffDiagonalLabelPairs Label, gY ≤ outcomePairLoss pair)
    (hX : ∀ pair ∈ b2V3OffDiagonalLabelPairs Label, gX ≤ treatmentPairLoss pair) :
    (gY + gX) / 2 ≤
      b2V3RestrictedGMin outcomePairLoss treatmentPairLoss hPairs := by
  unfold b2V3RestrictedGMin
  apply Finset.le_min'
  intro value hValue
  rcases Finset.mem_image.mp hValue with ⟨pair, hPair, rfl⟩
  exact div_le_div_of_nonneg_right (add_le_add (hY pair hPair) (hX pair hPair)) (by norm_num)

/-- At the X-null boundary, nonnegative X loss preserves only one half of the Y gap. -/
theorem b2V3RestrictedGMin_ge_half_outcome
    {Label : Type uLabel} [Fintype Label] [DecidableEq Label]
    (outcomePairLoss treatmentPairLoss : Label × Label → ℝ)
    (hPairs : (b2V3OffDiagonalLabelPairs Label).Nonempty)
    (gY : ℝ)
    (hY : ∀ pair ∈ b2V3OffDiagonalLabelPairs Label, gY ≤ outcomePairLoss pair)
    (hXNonneg : ∀ pair ∈ b2V3OffDiagonalLabelPairs Label,
      0 ≤ treatmentPairLoss pair) :
    gY / 2 ≤ b2V3RestrictedGMin outcomePairLoss treatmentPairLoss hPairs := by
  simpa using b2V3RestrictedGMin_ge_of_viewGaps outcomePairLoss treatmentPairLoss
    hPairs gY 0 hY hXNonneg

/-- Fusion pseudo-true mismatch loss calculated directly from the pair ledger. -/
def b2V3FusionPseudoTrueMismatchLoss
    {Candidate : Type uCandidate} {Label : Type uLabel}
    [Fintype Label] [DecidableEq Label]
    (mixingWeight : Candidate → Label × Label → ℝ)
    (outcomePairLoss treatmentPairLoss : Label × Label → ℝ)
    (candidate : Candidate) : ℝ :=
  b2V3PseudoTrueMismatchLoss mixingWeight
    (b2V3FusionPairLoss outcomePairLoss treatmentPairLoss) candidate

/-- Fusion mismatch is exactly the mean of the separately calculated Y and X mismatch losses. -/
theorem b2V3FusionPseudoTrueMismatchLoss_eq
    {Candidate : Type uCandidate} {Label : Type uLabel}
    [Fintype Label] [DecidableEq Label]
    (mixingWeight : Candidate → Label × Label → ℝ)
    (outcomePairLoss treatmentPairLoss : Label × Label → ℝ)
    (candidate : Candidate) :
    b2V3FusionPseudoTrueMismatchLoss mixingWeight outcomePairLoss treatmentPairLoss candidate =
      (b2V3PseudoTrueMismatchLoss mixingWeight outcomePairLoss candidate +
        b2V3PseudoTrueMismatchLoss mixingWeight treatmentPairLoss candidate) / 2 := by
  classical
  unfold b2V3FusionPseudoTrueMismatchLoss b2V3PseudoTrueMismatchLoss
    b2V3FusionPairLoss
  calc
    (∑ pair ∈ b2V3OffDiagonalLabelPairs Label,
        mixingWeight candidate pair * ((outcomePairLoss pair + treatmentPairLoss pair) / 2)) =
      ∑ pair ∈ b2V3OffDiagonalLabelPairs Label,
        (mixingWeight candidate pair * outcomePairLoss pair +
          mixingWeight candidate pair * treatmentPairLoss pair) / 2 := by
            apply Finset.sum_congr rfl
            intro pair _
            ring
    _ =
      ((∑ pair ∈ b2V3OffDiagonalLabelPairs Label,
          mixingWeight candidate pair * outcomePairLoss pair) +
        (∑ pair ∈ b2V3OffDiagonalLabelPairs Label,
          mixingWeight candidate pair * treatmentPairLoss pair)) / 2 := by
            simp only [add_div, Finset.sum_add_distrib, Finset.sum_div]

/-- A uniform lower bound on every fusion pair loss controls the complete mixing ledger. -/
theorem gMin_mul_b2V3TotalMixingMass_le_fusionPseudoTrueMismatchLoss
    {Candidate : Type uCandidate} {Label : Type uLabel}
    [Fintype Label] [DecidableEq Label]
    (mixingWeight : Candidate → Label × Label → ℝ)
    (outcomePairLoss treatmentPairLoss : Label × Label → ℝ)
    (candidate : Candidate) (gMin : ℝ)
    (hMixingNonneg : ∀ pair ∈ b2V3OffDiagonalLabelPairs Label,
      0 ≤ mixingWeight candidate pair)
    (hPairGap : ∀ pair ∈ b2V3OffDiagonalLabelPairs Label,
      gMin ≤ b2V3FusionPairLoss outcomePairLoss treatmentPairLoss pair) :
    gMin * b2V3TotalMixingMass mixingWeight candidate ≤
      b2V3FusionPseudoTrueMismatchLoss
        mixingWeight outcomePairLoss treatmentPairLoss candidate := by
  classical
  unfold b2V3TotalMixingMass b2V3FusionPseudoTrueMismatchLoss
    b2V3PseudoTrueMismatchLoss
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro pair hPair
  simpa [mul_comm] using
    (mul_le_mul_of_nonneg_left (hPairGap pair hPair)
      (hMixingNonneg pair hPair))

/-- Mean of two already normalized view risks; this is fusion, not a third raw component. -/
def b2V3FusionRisk
    {Candidate : Type uCandidate}
    (outcomeRisk treatmentRisk : Candidate → ℝ) (candidate : Candidate) : ℝ :=
  (outcomeRisk candidate + treatmentRisk candidate) / 2

/-- Mean of the normalized Y/X defects. -/
def b2V3FusionDefect (outcomeDefect treatmentDefect : ℝ) : ℝ :=
  (outcomeDefect + treatmentDefect) / 2

theorem b2V3FusionDefect_eq_mean (outcomeDefect treatmentDefect : ℝ) :
    b2V3FusionDefect outcomeDefect treatmentDefect =
      (outcomeDefect + treatmentDefect) / 2 := rfl

/-- Effective pair gap after absorbing a nonnegative defect on a positive impurity floor. -/
def b2V3NetPairGap (gMin defect minimumImpurity : ℝ) : ℝ :=
  gMin - defect / minimumImpurity

/-- Requested pure compatibility slope in normalized label distance. -/
def b2V3NetCompatibilitySlope
    (centerCount : ℕ) (gMin defect minimumImpurity : ℝ) : ℝ :=
  restrictedB2CompatibilitySlope centerCount
    (b2V3NetPairGap gMin defect minimumImpurity)

/-- Exact strict-margin condition in the source theorem's per-one-center scale. -/
def b2V3StrictMargin
    (centerCount : ℕ) (gMin defect minimumImpurity uY uX etaSel : ℝ) : Prop :=
  uY + uX + etaSel <
    restrictedB2Chi centerCount * b2V3NetPairGap gMin defect minimumImpurity

theorem b2V3StrictMargin_iff
    (centerCount : ℕ) (gMin defect minimumImpurity uY uX etaSel : ℝ) :
    b2V3StrictMargin centerCount gMin defect minimumImpurity uY uX etaSel ↔
      uY + uX + etaSel <
        restrictedB2Chi centerCount * (gMin - defect / minimumImpurity) := by
  rfl

/-- The requested margin is exactly the generic selector separation after `u_A2=(uY+uX)/2`. -/
theorem b2V3StrictMargin_implies_selector_separation
    (centerCount : ℕ) (gMin defect minimumImpurity uY uX etaSel : ℝ)
    (hCenterCount : centerCount = 9 ∨ centerCount = 12)
    (hMargin : b2V3StrictMargin centerCount gMin defect minimumImpurity uY uX etaSel) :
    2 * ((uY + uX) / 2) + etaSel <
      b2V3NetCompatibilitySlope centerCount gMin defect minimumImpurity /
        (centerCount : ℝ) := by
  have hCenterPos : (0 : ℝ) < centerCount := by
    rcases hCenterCount with rfl | rfl <;> norm_num
  unfold b2V3StrictMargin at hMargin
  unfold b2V3NetCompatibilitySlope restrictedB2CompatibilitySlope
  calc
    2 * ((uY + uX) / 2) + etaSel = uY + uX + etaSel := by ring
    _ < restrictedB2Chi centerCount * b2V3NetPairGap gMin defect minimumImpurity := hMargin
    _ = centerCount * restrictedB2Chi centerCount *
          b2V3NetPairGap gMin defect minimumImpurity / centerCount := by
      field_simp

/-- The restricted strict-margin parameter set is genuinely nonempty for both supported E values. -/
theorem b2V3StrictMarginRegion_nonempty
    (centerCount : ℕ) (hCenterCount : centerCount = 9 ∨ centerCount = 12) :
    ∃ gMin defect minimumImpurity uY uX etaSel : ℝ,
      0 < gMin ∧ 0 ≤ defect ∧ 0 < minimumImpurity ∧
      0 ≤ uY ∧ 0 ≤ uX ∧ 0 ≤ etaSel ∧
      b2V3StrictMargin centerCount gMin defect minimumImpurity uY uX etaSel := by
  refine ⟨1, 0, 1, 0, 0, 0, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, ?_⟩
  unfold b2V3StrictMargin b2V3NetPairGap
  have hChi := restrictedB2Chi_pos hCenterCount
  norm_num at hChi ⊢
  exact hChi

/-- Nonemptiness remains true after the A2 radii and a nonnegative selector tolerance are fixed. -/
theorem b2V3StrictMarginRegion_nonempty_for_fixed_radii
    (centerCount : ℕ) (hCenterCount : centerCount = 9 ∨ centerCount = 12)
    (uY uX etaSel : ℝ) (hUY : 0 ≤ uY) (hUX : 0 ≤ uX) (hEta : 0 ≤ etaSel) :
    ∃ gMin : ℝ, 0 < gMin ∧
      b2V3StrictMargin centerCount gMin 0 1 uY uX etaSel := by
  have hChi : 0 < restrictedB2Chi centerCount := restrictedB2Chi_pos hCenterCount
  let gMin := (uY + uX + etaSel) / restrictedB2Chi centerCount + 1
  refine ⟨gMin, ?_, ?_⟩
  · dsimp [gMin]
    have hNumerator : 0 ≤ uY + uX + etaSel := by linarith
    positivity
  · unfold b2V3StrictMargin b2V3NetPairGap
    dsimp [gMin]
    field_simp
    nlinarith

theorem b2V3NetPairGap_pos
    (gMin defect minimumImpurity : ℝ)
    (hMinimumImpurity : 0 < minimumImpurity)
    (hDominates : defect < gMin * minimumImpurity) :
    0 < b2V3NetPairGap gMin defect minimumImpurity := by
  unfold b2V3NetPairGap
  exact sub_pos.mpr ((div_lt_iff₀ hMinimumImpurity).2 hDominates)

theorem b2V3NetCompatibilitySlope_pos
    {Center : Type uCenter} [Fintype Center]
    (hCenterCount : Fintype.card Center = 9 ∨ Fintype.card Center = 12)
    (gMin defect minimumImpurity : ℝ)
    (hMinimumImpurity : 0 < minimumImpurity)
    (hDominates : defect < gMin * minimumImpurity) :
    0 < b2V3NetCompatibilitySlope (Fintype.card Center)
      gMin defect minimumImpurity := by
  exact restrictedB2CompatibilitySlope_pos hCenterCount
    (b2V3NetPairGap_pos gMin defect minimumImpurity hMinimumImpurity hDominates)

/--
Analytic restricted compatibility from separate X/Y pseudo-true decompositions.  The assumptions do
not contain the desired fusion risk-gap inequality.  Instead they expose the pair ledger, pair-gap
bound, per-view fitted-risk bridges, bad-partition impurity floor, and impurity/Hamming bridge.
The explicit exact-good and bad-record premises record the intended restricted domain; the
quantitative ledger assumptions, rather than those membership facts alone, prove the risk gap.
-/
theorem b2V3_primaryRisk_excess_ge_netSlope_mul_labelDistance
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [Fintype Center] [DecidableEq Center] [Nonempty Center]
    [Fintype Label] [DecidableEq Label]
    (truth : Partition Center Label)
    (good record : CandidateRecord Center Label Slot Context)
    (_hGood : good ∈ exactCandidateSet truth)
    (_hBad : record ∈ badCandidateSet truth)
    (mixingWeight : CandidateRecord Center Label Slot Context → Label × Label → ℝ)
    (outcomePairLoss treatmentPairLoss : Label × Label → ℝ)
    (outcomeRisk treatmentRisk : CandidateRecord Center Label Slot Context → ℝ)
    (outcomeDefect treatmentDefect gMin minimumImpurity : ℝ)
    (hOutcomeDefect : 0 ≤ outcomeDefect)
    (hTreatmentDefect : 0 ≤ treatmentDefect)
    (hMinimumImpurity : 0 < minimumImpurity)
    (hMixingNonneg : ∀ pair ∈ b2V3OffDiagonalLabelPairs Label,
      0 ≤ mixingWeight record pair)
    (hPairGap : ∀ pair ∈ b2V3OffDiagonalLabelPairs Label,
      gMin ≤ b2V3FusionPairLoss outcomePairLoss treatmentPairLoss pair)
    (hOutcomeBridge :
      b2V3PseudoTrueMismatchLoss mixingWeight outcomePairLoss record - outcomeDefect ≤
        outcomeRisk record - outcomeRisk good)
    (hTreatmentBridge :
      b2V3PseudoTrueMismatchLoss mixingWeight treatmentPairLoss record - treatmentDefect ≤
        treatmentRisk record - treatmentRisk good)
    (hMinimumMixing :
      minimumImpurity ≤ b2V3TotalMixingMass mixingWeight record)
    (hMixingSlope :
      (Fintype.card Center : ℝ) * restrictedB2Chi (Fintype.card Center) *
          labelDistance record.label truth ≤
        b2V3TotalMixingMass mixingWeight record)
    (hDominates :
      b2V3FusionDefect outcomeDefect treatmentDefect < gMin * minimumImpurity) :
    b2V3NetCompatibilitySlope (Fintype.card Center) gMin
          (b2V3FusionDefect outcomeDefect treatmentDefect) minimumImpurity *
        labelDistance record.label truth ≤
      b2V3FusionRisk outcomeRisk treatmentRisk record -
        b2V3FusionRisk outcomeRisk treatmentRisk good := by
  let defect := b2V3FusionDefect outcomeDefect treatmentDefect
  let mixing := b2V3TotalMixingMass mixingWeight record
  let netGap := b2V3NetPairGap gMin defect minimumImpurity
  have hDefect : 0 ≤ defect := by
    dsimp [defect, b2V3FusionDefect]
    positivity
  have hNetGap : 0 < netGap := by
    exact b2V3NetPairGap_pos gMin defect minimumImpurity hMinimumImpurity hDominates
  have hFusionPair :=
    gMin_mul_b2V3TotalMixingMass_le_fusionPseudoTrueMismatchLoss
      mixingWeight outcomePairLoss treatmentPairLoss record gMin
      hMixingNonneg hPairGap
  have hFusionBridge :
      b2V3FusionPseudoTrueMismatchLoss
          mixingWeight outcomePairLoss treatmentPairLoss record - defect ≤
        b2V3FusionRisk outcomeRisk treatmentRisk record -
          b2V3FusionRisk outcomeRisk treatmentRisk good := by
    rw [b2V3FusionPseudoTrueMismatchLoss_eq]
    dsimp [defect, b2V3FusionDefect, b2V3FusionRisk]
    linarith
  have hStructural :
      gMin * mixing - defect ≤
        b2V3FusionRisk outcomeRisk treatmentRisk record -
          b2V3FusionRisk outcomeRisk treatmentRisk good := by
    exact (sub_le_sub_right hFusionPair defect).trans hFusionBridge
  have hDefectScaled : defect ≤ defect / minimumImpurity * mixing := by
    calc
      defect = defect / minimumImpurity * minimumImpurity := by
        exact (div_mul_cancel₀ defect hMinimumImpurity.ne').symm
      _ ≤ defect / minimumImpurity * mixing := by
        exact mul_le_mul_of_nonneg_left hMinimumMixing
          (div_nonneg hDefect hMinimumImpurity.le)
  have hScaled := mul_le_mul_of_nonneg_left hMixingSlope hNetGap.le
  calc
    b2V3NetCompatibilitySlope (Fintype.card Center) gMin defect minimumImpurity *
          labelDistance record.label truth =
        netGap * ((Fintype.card Center : ℝ) *
          restrictedB2Chi (Fintype.card Center) *
            labelDistance record.label truth) := by
          dsimp [b2V3NetCompatibilitySlope, restrictedB2CompatibilitySlope, netGap]
          ring
    _ ≤ netGap * mixing := hScaled
    _ = gMin * mixing - defect / minimumImpurity * mixing := by
      dsimp [netGap, b2V3NetPairGap]
      ring
    _ ≤ gMin * mixing - defect := sub_le_sub_left hDefectScaled _
    _ ≤ b2V3FusionRisk outcomeRisk treatmentRisk record -
          b2V3FusionRisk outcomeRisk treatmentRisk good := hStructural

end

end BiostatFormal.HighDimensional.Multiview
