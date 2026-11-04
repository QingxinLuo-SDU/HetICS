import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.Objects
import Mathlib.Data.Fintype.Perm

/-!
# CAVES-C label recovery

The analysis label partition is compared with the latent partition only up to a permutation of
cluster names. No diagnostic alignment algorithm is part of this interface.
-/

namespace BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC

set_option autoImplicit false

variable {Center Cluster : Type*}

/-- Estimated center labels recover the latent labels up to a permutation of cluster names. -/
def LabelRecoveryUpToPermutation
    (trueLabel estimatedLabel : Center → Cluster) : Prop :=
  ∃ π : Equiv Cluster Cluster, ∀ c, estimatedLabel c = π (trueLabel c)

/-- Centers assigned to the analysis cluster named `k`. -/
def estimatedClusterCenters
    [Fintype Center] [DecidableEq Center] [DecidableEq Cluster]
    (estimatedLabel : Center → Cluster) (k : Cluster) : Finset Center :=
  Finset.univ.filter fun c => estimatedLabel c = k

/--
If `π` witnesses label recovery, analysis cluster `π k` contains exactly the centers whose latent
label is `k`.
-/
theorem estimatedClusterCenters_eq_true_of_witness
    [Fintype Center] [DecidableEq Center] [DecidableEq Cluster]
    (trueLabel estimatedLabel : Center → Cluster)
    (π : Equiv Cluster Cluster)
    (hWitness : ∀ c, estimatedLabel c = π (trueLabel c))
    (k : Cluster) :
    estimatedClusterCenters estimatedLabel (π k) =
      Finset.univ.filter fun c => trueLabel c = k := by
  ext c
  simp only [estimatedClusterCenters, Finset.mem_filter, Finset.mem_univ, true_and]
  rw [hWitness c]
  exact π.injective.eq_iff

end BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC
