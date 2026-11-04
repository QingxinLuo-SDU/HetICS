import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.Adjustment
import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.Labels
import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.WholeCenterPLRInference
import Mathlib.Tactic

/-!
# Learned-pipeline transport for CAVES-C

The actual labels, Y-side candidate rule, and strict stability filter are random because they are
learned from the data. This module supplies two non-interchangeable routes from that random pipeline
to a fixed qualified PLR reference:

* an exact route, on which the partition is recovered up to a single global permutation, the final
  adjustment sets agree, the parameter/fold/fitting context agrees, and both fits are usable;
* a first-order route, on which scaled point and variance perturbations vanish in measure.

Neither route claims that label recovery alone makes same-sample screening independent.
-/

open Filter MeasureTheory
open scoped Topology

namespace BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC

set_option autoImplicit false
noncomputable section

/-- The two fields consumed by cluster-specific PLR inference. -/
structure PLRReport where
  point : ℝ
  variance : ℝ
  deriving DecidableEq

/-- Centers in the same currently supplied label class as `c`. -/
def clusterMembers {Center Cluster : Type*}
    (label : Center → Cluster) : Center → Set Center :=
  fun c => {d | label d = label c}

/-- Equality of the induced partition is exactly what downstream clusterwise rules consume. -/
theorem clusterMembers_eq
    {Center Cluster : Type*} (a b : Center → Cluster)
    (hPartition : ∀ i j, a i = a j ↔ b i = b j) :
    clusterMembers a = clusterMembers b := by
  funext c
  ext d
  exact hPartition d c

/-- A single global relabeling witness preserves the entire center partition. -/
theorem partition_eq_of_labelRecovery
    {Center Cluster : Type*} (truth estimated : Center → Cluster)
    (hRecovery : LabelRecoveryUpToPermutation truth estimated) :
    ∀ i j, estimated i = estimated j ↔ truth i = truth j := by
  obtain ⟨permutation, witness⟩ := hRecovery
  intro i j
  rw [witness i, witness j]
  exact permutation.injective.eq_iff

/-- The whole learned rule and reference fit call, with folds explicit in the fit interface. -/
structure LearnedPLRPipeline
    (Context Center Cluster Fold Feature Parameter : Type*) [DecidableEq Feature] where
  ySide : Context → (Center → Set Center) → Center → Finset Feature
  strictStability :
    Context → (Center → Set Center) → Center → Finset Feature → Finset Feature
  fit : Context → Parameter → WholeCenterSplit Center Fold →
    (Center → Set Center) → (Center → Finset Feature) → Option PLRReport

variable {Context Center Cluster Fold Feature Parameter : Type*} [DecidableEq Feature]

/-- Final random adjustment sets after Y-side screening and strict stability filtering. -/
def learnedFinalSets
    (pipeline : LearnedPLRPipeline Context Center Cluster Fold Feature Parameter)
    (context : Context) (membership : Center → Set Center) : Center → Finset Feature :=
  fun c => pipeline.strictStability context membership c
    (pipeline.ySide context membership c)

/-- Full learned PLR fit for supplied labels. -/
def learnedPLR
    (pipeline : LearnedPLRPipeline Context Center Cluster Fold Feature Parameter)
    (context : Context) (parameter : Parameter) (split : WholeCenterSplit Center Fold)
    (label : Center → Cluster) : Option PLRReport :=
  pipeline.fit context parameter split (clusterMembers label)
    (learnedFinalSets pipeline context (clusterMembers label))

/-- Fixed qualified adjustment-set reference using the same fit implementation. -/
def referencePLR
    (pipeline : LearnedPLRPipeline Context Center Cluster Fold Feature Parameter)
    (context : Context) (parameter : Parameter) (split : WholeCenterSplit Center Fold)
    (truth : Center → Cluster) (qualified : Center → Finset Feature) : Option PLRReport :=
  pipeline.fit context parameter split (clusterMembers truth) qualified

/-- Learned state retained after the label and downstream-parameter stages complete. -/
structure LearnedPipelineState (Center Cluster Parameter : Type*) where
  label : Center → Cluster
  parameter : Parameter

/-- Unsuccessful label/parameter learning remains unavailable. -/
def actualPLR
    (pipeline : LearnedPLRPipeline Context Center Cluster Fold Feature Parameter)
    (context : Context) (split : WholeCenterSplit Center Fold)
    (state : Option (LearnedPipelineState Center Cluster Parameter)) : Option PLRReport :=
  state.bind fun learned => learnedPLR pipeline context learned.parameter split learned.label

/-- Global relabeling cannot change a learned fit that consumes only cluster memberships. -/
theorem learnedPLR_relabel
    (pipeline : LearnedPLRPipeline Context Center Cluster Fold Feature Parameter)
    (context : Context) (parameter : Parameter) (split : WholeCenterSplit Center Fold)
    (a b : Center → Cluster)
    (hPartition : ∀ i j, a i = a j ↔ b i = b j) :
    learnedPLR pipeline context parameter split a =
      learnedPLR pipeline context parameter split b := by
  unfold learnedPLR
  rw [clusterMembers_eq a b hPartition]

/-- Stability of the complete final rule, not merely label recovery, identifies the fixed
qualified reference fit. -/
theorem learnedPLR_eq_reference
    (pipeline : LearnedPLRPipeline Context Center Cluster Fold Feature Parameter)
    (context : Context) (parameter : Parameter) (split : WholeCenterSplit Center Fold)
    (truth : Center → Cluster) (qualified : Center → Finset Feature)
    (hStable : learnedFinalSets pipeline context (clusterMembers truth) = qualified) :
    learnedPLR pipeline context parameter split truth =
      referencePLR pipeline context parameter split truth qualified := by
  simp only [learnedPLR, referencePLR, hStable]

/-- Exact transport requires the same fitting context, parameter, and whole-center fold map in
addition to partition recovery and rule stability. -/
theorem actualPLR_eq_reference
    (pipeline : LearnedPLRPipeline Context Center Cluster Fold Feature Parameter)
    (contextActual contextReference : Context)
    (parameterActual parameterReference : Parameter)
    (splitActual splitReference : WholeCenterSplit Center Fold)
    (estimated truth : Center → Cluster) (qualified : Center → Finset Feature)
    (hContext : contextActual = contextReference)
    (hParameter : parameterActual = parameterReference)
    (hSplit : splitActual = splitReference)
    (hRecovery : LabelRecoveryUpToPermutation truth estimated)
    (hStable : learnedFinalSets pipeline contextReference
      (clusterMembers truth) = qualified) :
    actualPLR pipeline contextActual splitActual
        (some ⟨estimated, parameterActual⟩) =
      referencePLR pipeline contextReference parameterReference splitReference truth qualified := by
  subst contextActual
  subst parameterActual
  subst splitActual
  change learnedPLR pipeline contextReference parameterReference splitReference estimated = _
  exact (learnedPLR_relabel pipeline contextReference parameterReference splitReference
    estimated truth (partition_eq_of_labelRecovery truth estimated hRecovery)).trans
    (learnedPLR_eq_reference pipeline contextReference parameterReference splitReference
      truth qualified hStable)

/-- Event that the learned state exists and its labels recover the latent partition. -/
def learnedLabelEvent
    (state : Option (LearnedPipelineState Center Cluster Parameter))
    (truth : Center → Cluster) : Prop :=
  ∃ learned, state = some learned ∧
    LabelRecoveryUpToPermutation truth learned.label

/-- Event that the entire Y-side plus strict-stability output equals the fixed qualified rule. -/
def learnedRuleStableEvent
    (pipeline : LearnedPLRPipeline Context Center Cluster Fold Feature Parameter)
    (context : Context) (truth : Center → Cluster)
    (qualified : Center → Finset Feature) : Prop :=
  learnedFinalSets pipeline context (clusterMembers truth) = qualified

/-- Usability includes matching context, parameter, and fold map plus an available reference fit. -/
def learnedPLRUsableEvent
    (pipeline : LearnedPLRPipeline Context Center Cluster Fold Feature Parameter)
    (contextActual contextReference : Context)
    (parameterReference : Parameter)
    (splitActual splitReference : WholeCenterSplit Center Fold)
    (truth : Center → Cluster) (qualified : Center → Finset Feature)
    (state : Option (LearnedPipelineState Center Cluster Parameter)) : Prop :=
  contextActual = contextReference ∧ splitActual = splitReference ∧
    (∃ report, referencePLR pipeline contextReference parameterReference splitReference
      truth qualified = some report) ∧
    ∀ learned, state = some learned → learned.parameter = parameterReference

/-- The exact stability fields imply equality of the complete reports and actual availability. -/
theorem learned_fields_to_report
    (pipeline : LearnedPLRPipeline Context Center Cluster Fold Feature Parameter)
    (contextActual contextReference : Context)
    (parameterReference : Parameter)
    (splitActual splitReference : WholeCenterSplit Center Fold)
    (truth : Center → Cluster) (qualified : Center → Finset Feature)
    (state : Option (LearnedPipelineState Center Cluster Parameter))
    (hLabel : learnedLabelEvent state truth)
    (hStable : learnedRuleStableEvent pipeline contextReference truth qualified)
    (hUsable : learnedPLRUsableEvent pipeline contextActual contextReference parameterReference
      splitActual splitReference truth qualified state) :
    actualPLR pipeline contextActual splitActual state =
        referencePLR pipeline contextReference parameterReference splitReference truth qualified ∧
      actualPLR pipeline contextActual splitActual state ≠ none := by
  obtain ⟨learned, hState, hRecovery⟩ := hLabel
  obtain ⟨hContext, hSplit, ⟨report, hReference⟩, hParameter⟩ := hUsable
  have hEqual : actualPLR pipeline contextActual splitActual state =
      referencePLR pipeline contextReference parameterReference splitReference truth qualified := by
    rw [hState]
    exact actualPLR_eq_reference pipeline contextActual contextReference learned.parameter
      parameterReference splitActual splitReference learned.label truth qualified hContext
      (hParameter learned hState) hSplit hRecovery hStable
  exact ⟨hEqual, by rw [hEqual, hReference]; exact Option.some_ne_none report⟩

/-- Totalization is used only for unconditional limit arguments; availability remains explicit. -/
def extendPLRReport (fallback : PLRReport) (report : Option PLRReport) : PLRReport :=
  report.getD fallback

section Probability

variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)

/-- Label, Y-side, strict-stability, and usability failures are charged separately without
assuming independence. -/
theorem learned_four_failure_bound
    (labelGood ySideGood stabilityGood usableGood : Set Ω)
    (deltaLabel deltaYSide deltaStability deltaUsable : ENNReal)
    (hLabel : μ labelGoodᶜ ≤ deltaLabel)
    (hYSide : μ ySideGoodᶜ ≤ deltaYSide)
    (hStability : μ stabilityGoodᶜ ≤ deltaStability)
    (hUsable : μ usableGoodᶜ ≤ deltaUsable) :
    μ (labelGood ∩ ySideGood ∩ stabilityGood ∩ usableGood)ᶜ ≤
      deltaLabel + deltaYSide + deltaStability + deltaUsable := by
  simp only [Set.compl_inter]
  exact (measure_union_le _ _).trans
    (add_le_add
      ((measure_union_le _ _).trans
        (add_le_add
          ((measure_union_le _ _).trans (add_le_add hLabel hYSide))
          hStability))
      hUsable)

/-- Exact report equality on a high-probability event transfers both point and variance at their
required center-count scales and shows that unavailability vanishes. -/
theorem learned_reports_to_scaled_differences
    (actual reference : ℕ → Ω → Option PLRReport)
    (fallbackActual fallbackReference : ℕ → Ω → PLRReport)
    (pointScale varianceScale : ℕ → ℝ)
    (good : ℕ → Set Ω)
    (hFields : ∀ n ω, ω ∈ good n →
      actual n ω = reference n ω ∧ actual n ω ≠ none)
    (hGood : Tendsto (fun n => μ (good n)ᶜ) atTop (𝓝 0)) :
    TendstoInMeasure μ (fun n ω => pointScale n *
      ((extendPLRReport (fallbackActual n ω) (actual n ω)).point -
        (extendPLRReport (fallbackReference n ω) (reference n ω)).point)) atTop 0 ∧
    TendstoInMeasure μ (fun n ω => varianceScale n *
      ((extendPLRReport (fallbackActual n ω) (actual n ω)).variance -
        (extendPLRReport (fallbackReference n ω) (reference n ω)).variance)) atTop 0 ∧
    Tendsto (fun n => μ {ω | actual n ω = none}) atTop (𝓝 0) := by
  have hEqual : ∀ n ω, ω ∈ good n →
      extendPLRReport (fallbackActual n ω) (actual n ω) =
        extendPLRReport (fallbackReference n ω) (reference n ω) := by
    intro n ω hw
    obtain ⟨heq, hne⟩ := hFields n ω hw
    cases ha : actual n ω with
    | none => exact (hne ha).elim
    | some report => rw [← heq, ha]; rfl
  refine ⟨plr_scaled_difference_of_high_probability_eq μ _ _ pointScale good
      (fun n ω hw => congrArg PLRReport.point (hEqual n ω hw)) hGood,
    plr_scaled_difference_of_high_probability_eq μ _ _ varianceScale good
      (fun n ω hw => congrArg PLRReport.variance (hEqual n ω hw)) hGood, ?_⟩
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hGood
  · intro n
    exact bot_le
  · intro n
    apply measure_mono
    intro ω hNone hMembership
    exact (hFields n ω hMembership).2 hNone

/-- First-order alternative: explicit high-probability envelopes yield negligible point and
variance perturbations even when exact report equality is unavailable. -/
theorem learned_perturbation_transfer
    (pointActual pointReference varianceActual varianceReference : ℕ → Ω → ℝ)
    (pointScale varianceScale pointBound varianceBound : ℕ → ℝ)
    (good : ℕ → Set Ω)
    (hPoint : ∀ n ω, ω ∈ good n →
      |pointScale n * (pointActual n ω - pointReference n ω)| ≤ pointBound n)
    (hVariance : ∀ n ω, ω ∈ good n →
      |varianceScale n * (varianceActual n ω - varianceReference n ω)| ≤ varianceBound n)
    (hPointBound : Tendsto pointBound atTop (𝓝 0))
    (hVarianceBound : Tendsto varianceBound atTop (𝓝 0))
    (hGood : Tendsto (fun n => μ (good n)ᶜ) atTop (𝓝 0)) :
    TendstoInMeasure μ (fun n ω => pointScale n *
      (pointActual n ω - pointReference n ω)) atTop 0 ∧
    TendstoInMeasure μ (fun n ω => varianceScale n *
      (varianceActual n ω - varianceReference n ω)) atTop 0 :=
  ⟨plr_inMeasure_of_good_envelope μ _ pointBound good hPoint hPointBound hGood,
    plr_inMeasure_of_good_envelope μ _ varianceBound good hVariance
      hVarianceBound hGood⟩

end Probability

end

end BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC

