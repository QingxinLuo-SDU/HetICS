import ErrorTarget
import JointSandwich

/-! Increasing-center alternative to the fixed-center iid-row regime.

Each element of `Fin (n+1)` is one whole-center score contribution.  Dependence
inside a center is therefore unrestricted by this interface; the external CLT
is imposed across center contributions.  The full joint score covariance is
retained before extracting the person-weighted scalar target. -/

noncomputable section
open Matrix MeasureTheory Filter
open scoped BigOperators Topology

namespace BinaryFormal.CenterClusterInference

variable {C D K : Type*} [Fintype C] [Fintype D] [Fintype K]

/-- Empirical centering is across whole-center score vectors. -/
def centeredCenterScore (g : C -> D -> Real) (c : C) : D -> Real :=
  g c - JointSandwich.finiteMean Finset.univ g

/-- Scalar influence of a whole-center joint score after the common inverse,
target extraction, and person-count class weighting. -/
def centerInfluence (a : K -> Real) (H : Matrix K D Real)
    (R : Matrix D D Real) (g : C -> D -> Real) (c : C) : Real :=
  a ⬝ᵥ (H *ᵥ (R *ᵥ g c))

/-- Complete center-cluster sandwich; no diagonal or cross-coordinate deletion. -/
def centerReferenceVariance (M : Real) (a : K -> Real) (H : Matrix K D Real)
    (R : Matrix D D Real) (g : C -> D -> Real) : Real :=
  a ⬝ᵥ ((H * JointSandwich.covariance M R (centeredCenterScore g) * H.transpose) *ᵥ a)

/-- Exact finite identity behind the center-cluster variance. -/
theorem centerReferenceVariance_eq_sum (M : Real) (hM : 0 < M)
    (a : K -> Real) (H : Matrix K D Real) (R : Matrix D D Real)
    (g : C -> D -> Real) :
    centerReferenceVariance M a H R g =
      (M ^ 2)⁻¹ * ∑ c, (centerInfluence a H R (centeredCenterScore g) c) ^ 2 := by
  exact JointSandwich.sandwich_quadratic M hM R H a (centeredCenterScore g)

section Array

variable {Omega : Type*} [MeasurableSpace Omega]

/-- The center index at asymptotic scale `n` is `Fin (n+1)`; hence the square-root
rate below is the number-of-centers rate, not the number-of-persons rate. -/
def centerInfluenceArray
    (a : Nat -> K -> Real)
    (H : Nat -> Matrix K D Real)
    (R : Nat -> Matrix D D Real)
    (g : (n : Nat) -> Omega -> Fin (n+1) -> D -> Real) :
    (n : Nat) -> Omega -> Fin (n+1) -> Real :=
  fun n w c => centerInfluence (a n) (H n) (R n) (g n w) c

def centerReferenceVarianceArray
    (a : Nat -> K -> Real)
    (H : Nat -> Matrix K D Real)
    (R : Nat -> Matrix D D Real)
    (g : (n : Nat) -> Omega -> Fin (n+1) -> D -> Real) :
    Nat -> Omega -> Real :=
  fun n w => centerReferenceVariance ((n : Real)+1)
    (a n) (H n) (R n) (g n w)

/-- Conditional increasing-center theorem.  Root linearization, the independent
whole-center CLT, and consistency of the displayed center sandwich are explicit
inputs.  The theorem mechanically composes them with the full-sample selection
remainder and the actual reported variance. -/
theorem center_cluster_studentized
    (P : Measure Omega) [IsProbabilityMeasure P]
    (actual : Nat -> Omega -> Real) (target : Nat -> Real)
    (rM rSelection actualVariance : Nat -> Omega -> Real)
    (a : Nat -> K -> Real)
    (H : Nat -> Matrix K D Real)
    (R : Nat -> Matrix D D Real)
    (g : (n : Nat) -> Omega -> Fin (n+1) -> D -> Real)
    (sigma : Real) (hsigma : 0 < sigma)
    (expansion : ∀ n w, actual n w - target n =
      ErrorTarget.linearRowMean (centerInfluenceArray a H R g) n w +
        rM n w + rSelection n w)
    (centerCLT : TendstoInDistribution
      (fun (n : Nat) w => Real.sqrt ((n : Real)+1) *
        ErrorTarget.linearRowMean (centerInfluenceArray a H R g) n w / sigma)
      atTop id (fun _ => P) (ProbabilityTheory.gaussianReal 0 1))
    (remainder_small : TendstoInMeasure P
      (fun (n : Nat) w => Real.sqrt ((n : Real)+1) * rM n w / sigma) atTop 0)
    (selection_small : TendstoInMeasure P
      (fun (n : Nat) w => Real.sqrt ((n : Real)+1) * rSelection n w / sigma) atTop 0)
    (measurable_remainder : ∀ n, AEMeasurable (rM n) P)
    (measurable_selection : ∀ n, AEMeasurable (rSelection n) P)
    (center_variance_consistent : TendstoInMeasure P
      (fun (n : Nat) w => (n+1) * centerReferenceVarianceArray a H R g n w)
      atTop (fun _ => sigma^2))
    (actual_variance_close : TendstoInMeasure P
      (fun (n : Nat) w => (n+1) *
        (actualVariance n w - centerReferenceVarianceArray a H R g n w)) atTop 0)
    (measurable_variance : ∀ n, AEMeasurable (actualVariance n) P) :
    TendstoInDistribution
      (fun n w => (actual n w-target n) / Real.sqrt (actualVariance n w))
      atTop id (fun _ => P) (ProbabilityTheory.gaussianReal 0 1) := by
  exact ErrorTarget.actual_ate_studentized P actual target
    (centerInfluenceArray a H R g) rM rSelection
    (centerReferenceVarianceArray a H R g) actualVariance sigma hsigma
    expansion centerCLT remainder_small selection_small measurable_remainder
    measurable_selection center_variance_consistent actual_variance_close
    measurable_variance

end Array

#print axioms centeredCenterScore
#print axioms centerInfluence
#print axioms centerReferenceVariance
#print axioms centerReferenceVariance_eq_sum
#print axioms centerInfluenceArray
#print axioms centerReferenceVarianceArray
#print axioms center_cluster_studentized

end BinaryFormal.CenterClusterInference
