import FixedCenterM
noncomputable section
open Matrix FixedCenterM
open scoped BigOperators
open Filter MeasureTheory
open scoped Topology Matrix.Norms.Elementwise
namespace FixedCenterMConsumer

/-- Nonzero numerical residual: delta=1.5, score=2, residual=0.5. -/
theorem nonzero_solver_residual :
    (3/2:ℝ)-2 = (0:ℝ)-(1/2:ℝ) := by
  have h := linear_expansion (ContinuousLinearMap.id ℝ ℝ) (ContinuousLinearMap.id ℝ ℝ)
    (ContinuousLinearMap.id ℝ ℝ) (3/2:ℝ) 2 (1/2:ℝ) (fun x => rfl) (by norm_num)
  simpa using h

theorem zero_perturbation_means (u : Fin 2 → ℝ) : (avg u-avg u)^2 ≤ 0 := by
  exact (mean_perturbation_sq u u).trans (by simp [avg])

def twoCenterScore (c : Bool) (_ : Unit) (_ : Unit) : ℝ := if c then 1 else -1

/-- Global mean 0 does not remove the center-specific mean subtraction. -/
theorem fixed_center_counterexample :
    (∑ c : Bool, (1/2:ℝ)*avg (fun i : Unit => twoCenterScore c i ())) = 0 ∧
    actualMeat (fun _ : Bool => (1/2:ℝ)) twoCenterScore () () = 0 ∧
    (∑ c : Bool, (1/2:ℝ)*avg (fun i : Unit => (twoCenterScore c i ())^2)) = 1 := by
  norm_num [actualMeat, centeredMoment, avg, twoCenterScore, Fintype.sum_bool]

/-- A real derivative/bread consumer also returns the new actual centered-meat identity. -/
theorem actual_aipw_consumer {p I : Type*} [Fintype p] [DecidableEq p]
    [Fintype I] [Nonempty I] (r : I → p → ℝ) (x y : I → ℝ)
    (eta : Fin 3 → p → ℝ) (theta : ℝ)
    (K : Matrix (Fin 3 × p) (Fin 3 × p) ℝ)
    (hK : K*AIPWJacobian.meanInfo r x eta = 1) (i : I)
    (a b : (Fin 3 × p) ⊕ Unit) :
    JointSandwich.inverseCandidate K (AIPWJacobian.meanGrad r x y eta)*
      AIPWJacobian.actualBread r x y eta theta = 1 ∧
    centeredMoment (fun i => AIPWJacobian.joint (r i) (x i) (y i) eta theta) a b =
      avg (fun i => AIPWJacobian.joint (r i) (x i) (y i) eta theta a*
        AIPWJacobian.joint (r i) (x i) (y i) eta theta b)-
      avg (fun i => AIPWJacobian.joint (r i) (x i) (y i) eta theta a)*
        avg (fun i => AIPWJacobian.joint (r i) (x i) (y i) eta theta b) := by
  have hb := AIPWJacobian.actualBread_eq r x y eta theta
  have hi := AIPWJacobian.actual_inverse_and_contribution r x y eta theta K hK i
  exact ⟨hi.1, centeredMoment_eq _ a b⟩

/-- Actual row moments are fed into the sandwich theorem, not postulated as B consistency. -/
theorem actual_reference_variance {Ω C D : Type*} [MeasurableSpace Ω]
    [Fintype C] [Fintype D] [DecidableEq D] (μ : Measure Ω)
    {I : ℕ → C → Type*} [∀ n c, Fintype (I n c)] [∀ n c, Nonempty (I n c)]
    (w : ℕ → Ω → C → ℝ) (u v : (n : ℕ) → Ω → (c : C) → I n c → D → ℝ)
    (pi : C → ℝ) (m : C → D → ℝ) (Q : C → Matrix D D ℝ)
    (h : ℕ → Ω → D → ℝ) (Aref Apert : ℕ → Ω → Matrix D D ℝ)
    (h0 : D → ℝ) (A0 : Matrix D D ℝ) (hu : IsUnit A0.det)
    (hw : ∀ c, TendstoInMeasure μ (fun n ω => w n ω c) atTop (fun _ => pi c))
    (hm : ∀ c j, TendstoInMeasure μ (fun n ω => avg (fun i => u n ω c i j)) atTop (fun _ => m c j))
    (hq : ∀ c j k, TendstoInMeasure μ (fun n ω => avg (fun i => u n ω c i j*u n ω c i k)) atTop (fun _ => Q c j k))
    (hd : ∀ c j, TendstoInMeasure μ (fun n ω => avg (fun i => (v n ω c i j-u n ω c i j)^2)) atTop 0)
    (hh : TendstoInMeasure μ h atTop (fun _ => h0))
    (ha : TendstoInMeasure μ Aref atTop (fun _ => A0))
    (hap : TendstoInMeasure μ Apert atTop 0) :
    TendstoInMeasure μ (fun n ω => sandwichScalar (h n ω) ((Aref n ω+Apert n ω)⁻¹)
      (actualMeat (w n ω) (v n ω))) atTop
      (fun _ => sandwichScalar h0 A0⁻¹ (meatFromMoments pi m Q)) := by
  have hb := actualMeat_inMeasure μ w u v pi m Q hw hm hq hd
  exact actual_sandwich_inMeasure μ h Aref Apert _ h0 A0 _ hu hh ha hap hb

#print axioms nonzero_solver_residual
#print axioms zero_perturbation_means
#print axioms fixed_center_counterexample
#print axioms actual_aipw_consumer
#print axioms actual_reference_variance
end FixedCenterMConsumer
