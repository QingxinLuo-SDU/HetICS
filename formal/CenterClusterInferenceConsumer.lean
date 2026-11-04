import CenterClusterInference

noncomputable section
open BinaryFormal.CenterClusterInference

namespace BinaryFormal.CenterClusterInferenceConsumer

variable {C D K : Type*} [Fintype C] [Fintype D] [Fintype K]

/-- Consumer for the exact dense center-sandwich identity. -/
theorem dense_center_variance_consumed (M : Real) (hM : 0 < M)
    (a : K -> Real) (H : Matrix K D Real) (R : Matrix D D Real)
    (g : C -> D -> Real) :
    centerReferenceVariance M a H R g =
      (M ^ 2)⁻¹ * ∑ c, (centerInfluence a H R (centeredCenterScore g) c) ^ 2 :=
  centerReferenceVariance_eq_sum M hM a H R g

#print axioms dense_center_variance_consumed

end BinaryFormal.CenterClusterInferenceConsumer
