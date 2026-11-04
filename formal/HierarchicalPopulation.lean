import HierarchicalEffects
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability

/-! Common upper-level first/second moments for the cluster effects.

This is a conditional moment interface.  It does not prove iid sampling of
clusters, estimate the upper-level variance, or replace the represented-person
ATE by the upper-level location. -/

noncomputable section
open scoped BigOperators
open MeasureTheory

namespace BinaryFormal.HierarchicalPopulation

variable {Omega I K : Type*} [MeasurableSpace Omega] {P : Measure Omega}
  [DecidableEq I] [DecidableEq K] [Fintype K]

/-- Cluster effects have a common location and common finite second moment.
Additional iid/common-law assumptions remain an external statistical premise. -/
structure CommonUpperMomentModel (P : Measure Omega) (K : Type*) where
  location : Real
  deviation : K -> Omega -> Real
  variance : Real
  variance_nonneg : 0 <= variance
  deviation_integrable : ∀ k, Integrable (deviation k) P
  deviation_centered : ∀ k, ∫ w, deviation k w ∂P = 0
  deviation_sq_integrable : ∀ k, Integrable (fun w => (deviation k w) ^ 2) P
  common_second_moment : ∀ k, ∫ w, (deviation k w) ^ 2 ∂P = variance

def CommonUpperMomentModel.effect (m : CommonUpperMomentModel P K)
    (k : K) (w : Omega) : Real :=
  m.location + m.deviation k w

/-- Fixed person weights define the represented-population target in each
realization of the upper-level experiment. -/
def CommonUpperMomentModel.personTarget
    (m : CommonUpperMomentModel P K) (p : BinaryCore.SamplePartition I K)
    (w : Omega) : Real :=
  ∑ k, p.weight k * m.effect k w

omit [DecidableEq K] [Fintype K] in
theorem CommonUpperMomentModel.effect_integrable
    [IsProbabilityMeasure P] (m : CommonUpperMomentModel P K) (k : K) :
    Integrable (m.effect k) P := by
  exact (integrable_const m.location).add (m.deviation_integrable k)

theorem CommonUpperMomentModel.personTarget_integrable
    [IsProbabilityMeasure P] (m : CommonUpperMomentModel P K)
    (p : BinaryCore.SamplePartition I K) :
    Integrable (m.personTarget p) P := by
  unfold CommonUpperMomentModel.personTarget
  exact integrable_finsetSum Finset.univ (fun k _ =>
    (m.effect_integrable k).const_mul (p.weight k))

/-- With deterministic person-count weights, centered cluster deviations give
an upper-level expectation equal to the common location.  A realized weighted
ATE need not equal that location. -/
theorem CommonUpperMomentModel.integral_personTarget_eq_location
    [IsProbabilityMeasure P] (m : CommonUpperMomentModel P K)
    (p : BinaryCore.SamplePartition I K) :
    (∫ w, m.personTarget p w ∂P) = m.location := by
  unfold CommonUpperMomentModel.personTarget CommonUpperMomentModel.effect
  rw [integral_finsetSum Finset.univ]
  · simp_rw [integral_const_mul,
      integral_add (integrable_const m.location) (m.deviation_integrable _),
      m.deviation_centered]
    simp [← Finset.sum_mul, p.weights_sum_one]
  · intro k _
    exact ((integrable_const m.location).add (m.deviation_integrable k)).const_mul _

#print axioms CommonUpperMomentModel
#print axioms CommonUpperMomentModel.effect
#print axioms CommonUpperMomentModel.personTarget
#print axioms CommonUpperMomentModel.effect_integrable
#print axioms CommonUpperMomentModel.personTarget_integrable
#print axioms CommonUpperMomentModel.integral_personTarget_eq_location

end BinaryFormal.HierarchicalPopulation
