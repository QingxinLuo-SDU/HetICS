import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

/-!
# Abstract probability stitching for multiview recovery

This file formalizes only the first-failure event algebra behind `(MV22a)` and `(A.33)`.  The
events are abstract subsets of one sample space.  No independence, conditional probability,
candidate coverage, risk deviation, compatibility, refinement, or B2-specific constant is used.
-/

open MeasureTheory

namespace BiostatFormal.HighDimensional.Multiview

universe uΩ

/--
The one-charge union of the first failures in the fixed order
`coverage → refinement survival → deviation → gap → output`.
-/
def firstFailureUnion {Ω : Type uΩ} (C R D G O : Set Ω) : Set Ω :=
  Cᶜ ∪
    (C ∩ Rᶜ) ∪
    (C ∩ R ∩ Dᶜ) ∪
    (C ∩ R ∩ D ∩ Gᶜ) ∪
    (C ∩ R ∩ D ∩ G ∩ Oᶜ)

/--
If simultaneous success of `C`, `R`, `D`, `G`, and `O` implies exact recovery `Z`, then every
failure of exact recovery belongs to one of the five ordered first-failure branches.
-/
theorem failure_subset_first_failure_union
    {Ω : Type uΩ} {C R D G O Z : Set Ω}
    (hSuccess : C ∩ R ∩ D ∩ G ∩ O ⊆ Z) :
    Zᶜ ⊆ firstFailureUnion C R D G O := by
  intro ω hZ
  by_cases hC : ω ∈ C
  · by_cases hR : ω ∈ R
    · by_cases hD : ω ∈ D
      · by_cases hG : ω ∈ G
        · by_cases hO : ω ∈ O
          · exact (hZ (hSuccess ⟨⟨⟨⟨hC, hR⟩, hD⟩, hG⟩, hO⟩)).elim
          · simp [firstFailureUnion, hC, hR, hD, hG, hO]
        · simp [firstFailureUnion, hC, hR, hD, hG]
      · simp [firstFailureUnion, hC, hR, hD]
    · simp [firstFailureUnion, hC, hR]
  · simp [firstFailureUnion, hC]

/--
Five unconditional cross-event bounds give an upper bound on exact-recovery failure.  This is a
finite union bound on one joint measure; no conditional probability or independence is assumed.
-/
theorem measure_exactRecoveryFailure_le
    {Ω : Type uΩ} [MeasurableSpace Ω]
    (μ : Measure Ω) {C R D G O Z : Set Ω}
    {δC δR δD δG δO : ENNReal}
    (hSuccess : C ∩ R ∩ D ∩ G ∩ O ⊆ Z)
    (hC : μ Cᶜ ≤ δC)
    (hR : μ (C ∩ Rᶜ) ≤ δR)
    (hD : μ (C ∩ R ∩ Dᶜ) ≤ δD)
    (hG : μ (C ∩ R ∩ D ∩ Gᶜ) ≤ δG)
    (hO : μ (C ∩ R ∩ D ∩ G ∩ Oᶜ) ≤ δO) :
    μ Zᶜ ≤ δC + δR + δD + δG + δO := by
  calc
    μ Zᶜ ≤ μ (firstFailureUnion C R D G O) :=
      measure_mono (failure_subset_first_failure_union hSuccess)
    _ ≤
        μ Cᶜ + μ (C ∩ Rᶜ) + μ (C ∩ R ∩ Dᶜ) +
          μ (C ∩ R ∩ D ∩ Gᶜ) + μ (C ∩ R ∩ D ∩ G ∩ Oᶜ) := by
      unfold firstFailureUnion
      calc
        μ (Cᶜ ∪ (C ∩ Rᶜ) ∪ (C ∩ R ∩ Dᶜ) ∪
              (C ∩ R ∩ D ∩ Gᶜ) ∪ (C ∩ R ∩ D ∩ G ∩ Oᶜ))
            ≤ μ (Cᶜ ∪ (C ∩ Rᶜ) ∪ (C ∩ R ∩ Dᶜ) ∪
                (C ∩ R ∩ D ∩ Gᶜ)) + μ (C ∩ R ∩ D ∩ G ∩ Oᶜ) :=
          measure_union_le _ _
        _ ≤
            (μ (Cᶜ ∪ (C ∩ Rᶜ) ∪ (C ∩ R ∩ Dᶜ)) +
              μ (C ∩ R ∩ D ∩ Gᶜ)) + μ (C ∩ R ∩ D ∩ G ∩ Oᶜ) := by
          gcongr
          exact measure_union_le _ _
        _ ≤
            ((μ (Cᶜ ∪ (C ∩ Rᶜ)) + μ (C ∩ R ∩ Dᶜ)) +
              μ (C ∩ R ∩ D ∩ Gᶜ)) + μ (C ∩ R ∩ D ∩ G ∩ Oᶜ) := by
          gcongr
          exact measure_union_le _ _
        _ ≤
            (((μ Cᶜ + μ (C ∩ Rᶜ)) + μ (C ∩ R ∩ Dᶜ)) +
              μ (C ∩ R ∩ D ∩ Gᶜ)) + μ (C ∩ R ∩ D ∩ G ∩ Oᶜ) := by
          gcongr
          exact measure_union_le _ _
    _ ≤ δC + δR + δD + δG + δO :=
      add_le_add (add_le_add (add_le_add (add_le_add hC hR) hD) hG) hO

/--
Under a probability measure, the failure upper bound yields the corresponding recovery lower
bound.  Only measurability of the recovery event is needed for the complement identity.
-/
theorem probability_exactRecovery_ge
    {Ω : Type uΩ} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {C R D G O Z : Set Ω} {δC δR δD δG δO : ENNReal}
    (hZ : MeasurableSet Z)
    (hSuccess : C ∩ R ∩ D ∩ G ∩ O ⊆ Z)
    (hC : μ Cᶜ ≤ δC)
    (hR : μ (C ∩ Rᶜ) ≤ δR)
    (hD : μ (C ∩ R ∩ Dᶜ) ≤ δD)
    (hG : μ (C ∩ R ∩ D ∩ Gᶜ) ≤ δG)
    (hO : μ (C ∩ R ∩ D ∩ G ∩ Oᶜ) ≤ δO) :
    1 - (δC + δR + δD + δG + δO) ≤ μ Z := by
  have hFailure : μ Zᶜ ≤ δC + δR + δD + δG + δO :=
    measure_exactRecoveryFailure_le μ hSuccess hC hR hD hG hO
  have hRecovery : μ Z = 1 - μ Zᶜ := by
    calc
      μ Z = μ (Zᶜ)ᶜ := by rw [compl_compl]
      _ = μ Set.univ - μ Zᶜ := measure_compl hZ.compl (measure_ne_top μ Zᶜ)
      _ = 1 - μ Zᶜ := by simp
  calc
    1 - (δC + δR + δD + δG + δO) ≤ 1 - μ Zᶜ :=
      tsub_le_tsub_left hFailure 1
    _ = μ Z := hRecovery.symm

end BiostatFormal.HighDimensional.Multiview
