import HierarchicalPopulation

noncomputable section
open MeasureTheory
open BinaryFormal.HierarchicalPopulation

namespace BinaryFormal.HierarchicalPopulationConsumer

variable {Omega I K : Type*} [MeasurableSpace Omega] {P : Measure Omega}
  [DecidableEq I] [DecidableEq K] [Fintype K]

/-- Public consumer: the common-moment model reaches the person-weighted
expectation without identifying a realized target with the location. -/
theorem common_location_is_expected_person_target
    (P : Measure Omega) [IsProbabilityMeasure P]
    (m : CommonUpperMomentModel P K) (p : BinaryCore.SamplePartition I K) :
    (∫ w, m.personTarget p w ∂P) = m.location :=
  m.integral_personTarget_eq_location p

#print axioms common_location_is_expected_person_target

end BinaryFormal.HierarchicalPopulationConsumer
