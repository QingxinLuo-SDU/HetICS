import BinaryCore
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability

/-! Minimal formal interface for the paper's hierarchical causal model.

The general person-weighted identity remains unchanged.  This module adds the
application restriction that row/center targets are constant inside a latent
cluster, while cluster mechanisms and cluster effects may differ.  It does not
assert that the Python DGP satisfies the restriction.  The companion module
`HierarchicalPopulation` makes
the common first/second-moment interface explicit and proves the expectation of
the fixed person-weighted target.  Independence, identical distribution beyond
those moments, and random-effects estimation are separate statistical inputs. -/

noncomputable section
open scoped BigOperators
open MeasureTheory

namespace BinaryFormal.HierarchicalEffects

variable {I K V : Type*} [DecidableEq I] [DecidableEq K]

/-- A cluster may have its own active covariates and its own risk-difference ATE. -/
structure ClusterMechanism (V : Type*) [DecidableEq V] where
  active : Finset V
  effect : ℝ

/-- Every represented row/center in one latent cluster has that cluster's target effect. -/
def WithinClusterStable (p : BinaryCore.SamplePartition I K)
    (rowEffect : I → ℝ) (clusterEffect : K → ℝ) : Prop :=
  ∀ i ∈ p.S, rowEffect i = clusterEffect (p.cluster i)

namespace WithinClusterStable

/-- Within-cluster stability turns the empirical fiber mean of represented effects
into the corresponding cluster target. -/
theorem clusterMean_eq (p : BinaryCore.SamplePartition I K)
    (rowEffect : I → ℝ) (clusterEffect : K → ℝ)
    (h : WithinClusterStable p rowEffect clusterEffect) (k : K) :
    p.clusterMean rowEffect k = clusterEffect k := by
  have hn : (p.n k : ℝ) ≠ 0 := by
    exact_mod_cast (ne_of_gt (p.n_pos k))
  unfold BinaryCore.SamplePartition.clusterMean
  apply (div_eq_iff hn).2
  calc
    ∑ i ∈ p.fiber k, rowEffect i =
        ∑ i ∈ p.fiber k, clusterEffect k := by
          apply Finset.sum_congr rfl
          intro i hi
          have hiS : i ∈ p.S := (Finset.mem_filter.mp hi).1
          have hik : p.cluster i = k := (Finset.mem_filter.mp hi).2
          simpa [hik] using h i hiS
    _ = clusterEffect k * (p.n k : ℝ) := by
          simp [BinaryCore.SamplePartition.n, mul_comm]

/-- With actual person-count weights, stable cluster effects aggregate to the
full represented-population mean. -/
theorem personWeighted_eq_fullMean (p : BinaryCore.SamplePartition I K)
    [Fintype K]
    (rowEffect : I → ℝ) (clusterEffect : K → ℝ)
    (h : WithinClusterStable p rowEffect clusterEffect) :
    ∑ k, p.weight k * clusterEffect k = p.fullMean rowEffect := by
  calc
    ∑ k, p.weight k * clusterEffect k =
        ∑ k, p.weight k * p.clusterMean rowEffect k := by
          apply Finset.sum_congr rfl
          intro k _
          rw [h.clusterMean_eq p rowEffect clusterEffect k]
    _ = p.fullMean rowEffect := p.weighted_mean_eq_full rowEffect

end WithinClusterStable

variable [Fintype K]

/-- Algebraic random-effects interface: person weights preserve the common
upper-level location and aggregate the supplied cluster deviations. -/
theorem weighted_decomposition (p : BinaryCore.SamplePartition I K)
    (mu : ℝ) (deviation : K → ℝ) :
    ∑ k, p.weight k * (mu + deviation k) =
      mu + ∑ k, p.weight k * deviation k := by
  calc
    ∑ k, p.weight k * (mu + deviation k) =
        ∑ k, (p.weight k * mu + p.weight k * deviation k) := by
          apply Finset.sum_congr rfl
          intro k _
          ring
    _ = (∑ k, p.weight k * mu) + ∑ k, p.weight k * deviation k := by
          rw [Finset.sum_add_distrib]
    _ = mu + ∑ k, p.weight k * deviation k := by
          rw [← Finset.sum_mul, p.weights_sum_one, one_mul]

/-- A zero person-weighted deviation identifies the upper-level location with
the person-weighted cluster target. -/
theorem weighted_eq_upperMean (p : BinaryCore.SamplePartition I K)
    (mu : ℝ) (deviation : K → ℝ)
    (hzero : ∑ k, p.weight k * deviation k = 0) :
    ∑ k, p.weight k * (mu + deviation k) = mu := by
  rw [weighted_decomposition p mu deviation, hzero, add_zero]

#print axioms ClusterMechanism
#print axioms WithinClusterStable
#print axioms WithinClusterStable.clusterMean_eq
#print axioms WithinClusterStable.personWeighted_eq_fullMean
#print axioms weighted_decomposition
#print axioms weighted_eq_upperMean

end BinaryFormal.HierarchicalEffects
