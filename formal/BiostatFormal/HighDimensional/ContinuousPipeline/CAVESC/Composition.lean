import Mathlib.MeasureTheory.Measure.MeasureSpaceDef
import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.Labels
import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.Adjustment
import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.WholeCenterPLR
import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.RandomEffectsGLS

/-!
# Conditional end-to-end composition for CAVES-C

This module composes six explicit success events and charges output failure to the first failed
stage. It neither manufactures any stage certificate nor assumes independence. In particular,
expected rejection-FDR accounting remains a separate ledger: it is not a simultaneous
core-retention event and is not one of the six success fields below.

The measure result is an output-failure upper bound obtained only from set inclusion and finite
subadditivity. It is not a confidence-interval coverage theorem.
-/

open MeasureTheory

namespace BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC

set_option autoImplicit false

/--
Six explicit events required by the conditional CAVES-C theory chain. None is constructed by this
structure; each field must be supplied by the caller for the realized outcome `ω`.
-/
structure CAVESCCertificates {Ω : Type*}
    (labelGood ySideCoreGood stabilityCoreGood adjustmentGood plrGood aggregationGood : Set Ω)
    (ω : Ω) : Prop where
  /-- Label recovery up to one global permutation of latent-cluster names. -/
  labels : ω ∈ labelGood
  /-- Ideal Y-side core detectability and retention into the candidate set. -/
  ySide : ω ∈ ySideCoreGood
  /-- Simultaneous strict non-rejection/core retention, not rejection-FDR control. -/
  stability : ω ∈ stabilityCoreGood
  /-- Safe adjustment together with caller-supplied identification, positivity, and stability. -/
  adjustment : ω ∈ adjustmentGood
  /-- Whole-center PLR and its external nuisance-rate, center-CLT, and covariance conditions. -/
  plr : ω ∈ plrGood
  /-- Covariance-aware REML/GLS and its external covariance, REML, and interval conditions. -/
  aggregation : ω ∈ aggregationGood

/--
Conditional deterministic composition: an explicit bridge maps all six supplied stage
certificates to the requested output event. This theorem creates no stage certificate itself.
-/
theorem cavesC_endToEnd_conditional
    {Ω : Type*}
    {labelGood ySideCoreGood stabilityCoreGood adjustmentGood plrGood aggregationGood outputGood :
      Set Ω}
    (hBridge : ∀ ω,
      CAVESCCertificates labelGood ySideCoreGood stabilityCoreGood
        adjustmentGood plrGood aggregationGood ω →
      ω ∈ outputGood)
    {ω : Ω}
    (certificate : CAVESCCertificates labelGood ySideCoreGood stabilityCoreGood
      adjustmentGood plrGood aggregationGood ω) :
    ω ∈ outputGood :=
  hBridge ω certificate

/-- The event on which all six named CAVES-C stages succeed. -/
def sixStageSuccess {Ω : Type*}
    (labelGood ySideCoreGood stabilityCoreGood adjustmentGood plrGood aggregationGood : Set Ω) :
    Set Ω :=
  (((((labelGood ∩ ySideCoreGood) ∩ stabilityCoreGood) ∩ adjustmentGood) ∩ plrGood) ∩
    aggregationGood)

/-- First failure at stage 1: label recovery fails. -/
def labelFirstFailure {Ω : Type*} (labelGood : Set Ω) : Set Ω := labelGoodᶜ

/-- First failure at stage 2: labels succeed but ideal Y-side core retention fails. -/
def ySideFirstFailure {Ω : Type*} (labelGood ySideCoreGood : Set Ω) : Set Ω :=
  labelGood ∩ ySideCoreGoodᶜ

/-- First failure at stage 3: the core is not simultaneously retained by strict stability. -/
def stabilityFirstFailure {Ω : Type*}
    (labelGood ySideCoreGood stabilityCoreGood : Set Ω) : Set Ω :=
  (labelGood ∩ ySideCoreGood) ∩ stabilityCoreGoodᶜ

/-- First failure at stage 4: safe adjustment and its scientific conditions fail. -/
def adjustmentFirstFailure {Ω : Type*}
    (labelGood ySideCoreGood stabilityCoreGood adjustmentGood : Set Ω) : Set Ω :=
  ((labelGood ∩ ySideCoreGood) ∩ stabilityCoreGood) ∩ adjustmentGoodᶜ

/-- First failure at stage 5: whole-center PLR and its external conditions fail. -/
def plrFirstFailure {Ω : Type*}
    (labelGood ySideCoreGood stabilityCoreGood adjustmentGood plrGood : Set Ω) : Set Ω :=
  (((labelGood ∩ ySideCoreGood) ∩ stabilityCoreGood) ∩ adjustmentGood) ∩ plrGoodᶜ

/-- First failure at stage 6: covariance-aware REML/GLS and its external conditions fail. -/
def aggregationFirstFailure {Ω : Type*}
    (labelGood ySideCoreGood stabilityCoreGood adjustmentGood plrGood aggregationGood : Set Ω) :
    Set Ω :=
  ((((labelGood ∩ ySideCoreGood) ∩ stabilityCoreGood) ∩ adjustmentGood) ∩ plrGood) ∩
    aggregationGoodᶜ

/--
Left-associated union of the six ordered first-failure events: labels, Y-side core retention,
strict stability retention, safe adjustment, whole-center PLR, then covariance-aware aggregation.
The separate expected rejection-FDR ledger is intentionally absent.
-/
def sixStageFailureUnion {Ω : Type*}
    (labelGood ySideCoreGood stabilityCoreGood adjustmentGood plrGood aggregationGood : Set Ω) :
    Set Ω :=
  (((((labelFirstFailure labelGood ∪ ySideFirstFailure labelGood ySideCoreGood) ∪
      stabilityFirstFailure labelGood ySideCoreGood stabilityCoreGood) ∪
      adjustmentFirstFailure labelGood ySideCoreGood stabilityCoreGood adjustmentGood) ∪
      plrFirstFailure labelGood ySideCoreGood stabilityCoreGood adjustmentGood plrGood) ∪
      aggregationFirstFailure labelGood ySideCoreGood stabilityCoreGood adjustmentGood plrGood
        aggregationGood)

/-- Every output failure is charged to the first failed event among the six ordered stages. -/
theorem outputFailure_subset_sixStageFailureUnion
    {Ω : Type*}
    {labelGood ySideCoreGood stabilityCoreGood adjustmentGood plrGood aggregationGood outputGood :
      Set Ω}
    (hSuccess : sixStageSuccess labelGood ySideCoreGood stabilityCoreGood adjustmentGood plrGood
      aggregationGood ⊆ outputGood) :
    outputGoodᶜ ⊆ sixStageFailureUnion labelGood ySideCoreGood stabilityCoreGood
      adjustmentGood plrGood aggregationGood := by
  intro ω hOutput
  simp only [sixStageFailureUnion, Set.mem_union]
  by_cases hLabel : ω ∈ labelGood
  · by_cases hYSide : ω ∈ ySideCoreGood
    · by_cases hStability : ω ∈ stabilityCoreGood
      · by_cases hAdjustment : ω ∈ adjustmentGood
        · by_cases hPLR : ω ∈ plrGood
          · by_cases hAggregation : ω ∈ aggregationGood
            · exact (hOutput (hSuccess (by
                simp [sixStageSuccess, hLabel, hYSide, hStability, hAdjustment, hPLR,
                  hAggregation]))).elim
            · apply Or.inr
              exact ⟨⟨⟨⟨⟨hLabel, hYSide⟩, hStability⟩, hAdjustment⟩, hPLR⟩, hAggregation⟩
          · apply Or.inl
            apply Or.inr
            exact ⟨⟨⟨⟨hLabel, hYSide⟩, hStability⟩, hAdjustment⟩, hPLR⟩
        · apply Or.inl
          apply Or.inl
          apply Or.inr
          exact ⟨⟨⟨hLabel, hYSide⟩, hStability⟩, hAdjustment⟩
      · apply Or.inl
        apply Or.inl
        apply Or.inl
        apply Or.inr
        exact ⟨⟨hLabel, hYSide⟩, hStability⟩
    · apply Or.inl
      apply Or.inl
      apply Or.inl
      apply Or.inl
      apply Or.inr
      exact ⟨hLabel, hYSide⟩
  · apply Or.inl
    apply Or.inl
    apply Or.inl
    apply Or.inl
    apply Or.inl
    exact hLabel

/--
Six ordered first-failure budgets bound output failure by finite subadditivity alone. No event
independence, mutual exclusivity, stage-event measurability, or probability-measure assumption is
used. This is not an interval-coverage theorem.
-/
theorem measure_sixStagePipelineFailure_le
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    {labelGood ySideCoreGood stabilityCoreGood adjustmentGood plrGood aggregationGood outputGood :
      Set Ω}
    {δLabel δYSide δStability δAdjustment δPLR δAggregation : ENNReal}
    (hSuccess : sixStageSuccess labelGood ySideCoreGood stabilityCoreGood adjustmentGood plrGood
      aggregationGood ⊆ outputGood)
    (hLabel : μ labelGoodᶜ ≤ δLabel)
    (hYSide : μ (labelGood ∩ ySideCoreGoodᶜ) ≤ δYSide)
    (hStability : μ ((labelGood ∩ ySideCoreGood) ∩ stabilityCoreGoodᶜ) ≤ δStability)
    (hAdjustment :
      μ (((labelGood ∩ ySideCoreGood) ∩ stabilityCoreGood) ∩ adjustmentGoodᶜ) ≤ δAdjustment)
    (hPLR :
      μ ((((labelGood ∩ ySideCoreGood) ∩ stabilityCoreGood) ∩ adjustmentGood) ∩ plrGoodᶜ) ≤
        δPLR)
    (hAggregation :
      μ (((((labelGood ∩ ySideCoreGood) ∩ stabilityCoreGood) ∩ adjustmentGood) ∩ plrGood) ∩
        aggregationGoodᶜ) ≤ δAggregation) :
    μ outputGoodᶜ ≤
      δLabel + δYSide + δStability + δAdjustment + δPLR + δAggregation := by
  let first : Set Ω := labelGoodᶜ
  let second : Set Ω := labelGood ∩ ySideCoreGoodᶜ
  let third : Set Ω := (labelGood ∩ ySideCoreGood) ∩ stabilityCoreGoodᶜ
  let fourth : Set Ω :=
    ((labelGood ∩ ySideCoreGood) ∩ stabilityCoreGood) ∩ adjustmentGoodᶜ
  let fifth : Set Ω :=
    (((labelGood ∩ ySideCoreGood) ∩ stabilityCoreGood) ∩ adjustmentGood) ∩ plrGoodᶜ
  let sixth : Set Ω :=
    ((((labelGood ∩ ySideCoreGood) ∩ stabilityCoreGood) ∩ adjustmentGood) ∩ plrGood) ∩
      aggregationGoodᶜ
  have hSecondUnion : μ (first ∪ second) ≤ μ first + μ second := measure_union_le _ _
  have hThirdUnion : μ ((first ∪ second) ∪ third) ≤ μ (first ∪ second) + μ third :=
    measure_union_le _ _
  have hFourthUnion :
      μ (((first ∪ second) ∪ third) ∪ fourth) ≤
        μ ((first ∪ second) ∪ third) + μ fourth :=
    measure_union_le _ _
  have hFifthUnion :
      μ ((((first ∪ second) ∪ third) ∪ fourth) ∪ fifth) ≤
        μ (((first ∪ second) ∪ third) ∪ fourth) + μ fifth :=
    measure_union_le _ _
  have hSixthUnion :
      μ (((((first ∪ second) ∪ third) ∪ fourth) ∪ fifth) ∪ sixth) ≤
        μ ((((first ∪ second) ∪ third) ∪ fourth) ∪ fifth) + μ sixth :=
    measure_union_le _ _
  calc
    μ outputGoodᶜ ≤ μ (sixStageFailureUnion labelGood ySideCoreGood stabilityCoreGood
        adjustmentGood plrGood aggregationGood) :=
      measure_mono (outputFailure_subset_sixStageFailureUnion hSuccess)
    _ = μ (((((first ∪ second) ∪ third) ∪ fourth) ∪ fifth) ∪ sixth) := by
      rfl
    _ ≤ μ ((((first ∪ second) ∪ third) ∪ fourth) ∪ fifth) + μ sixth := hSixthUnion
    _ ≤ (μ (((first ∪ second) ∪ third) ∪ fourth) + μ fifth) + μ sixth := by
      exact add_le_add_left hFifthUnion _
    _ ≤ ((μ ((first ∪ second) ∪ third) + μ fourth) + μ fifth) + μ sixth := by
      exact add_le_add_left (add_le_add_left hFourthUnion _) _
    _ ≤ (((μ (first ∪ second) + μ third) + μ fourth) + μ fifth) + μ sixth := by
      exact add_le_add_left (add_le_add_left (add_le_add_left hThirdUnion _) _) _
    _ ≤ ((((μ first + μ second) + μ third) + μ fourth) + μ fifth) + μ sixth := by
      exact add_le_add_left
        (add_le_add_left (add_le_add_left (add_le_add_left hSecondUnion _) _) _) _
    _ ≤ ((((δLabel + δYSide) + δStability) + δAdjustment) + δPLR) + δAggregation := by
      exact add_le_add
        (add_le_add (add_le_add (add_le_add (add_le_add hLabel hYSide) hStability)
          hAdjustment) hPLR) hAggregation
    _ = δLabel + δYSide + δStability + δAdjustment + δPLR + δAggregation := rfl

end BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC
