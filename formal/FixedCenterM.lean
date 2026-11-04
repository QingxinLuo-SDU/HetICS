import ErrorTarget
import PersonWeightedJoint
import Mathlib.Analysis.Normed.Ring.Units
import Mathlib.Topology.Instances.Matrix
import Mathlib.Analysis.Matrix.Normed

/-! Deterministic mechanisms for fixed-center M estimation. Statistical LLN/CLT
inputs remain explicit; neither root consistency nor final variance consistency
is a field of a purported regularity structure. -/
noncomputable section
open Filter MeasureTheory Matrix
open scoped Topology BigOperators
open scoped Matrix.Norms.Elementwise
namespace FixedCenterM
set_option maxHeartbeats 200000

section Root
variable {E F : Type*} [NormedAddCommGroup E] [NormedAddCommGroup F]

/-- Uniform empirical error and numerical residual locate the selected root. -/
theorem root_localization (U Un : E → F) (root estimate : E) (eps gap err res : ℝ)
    (sep : ∀ x, eps ≤ ‖x-root‖ → gap ≤ ‖U x‖)
    (uniform : ∀ x, ‖Un x-U x‖ ≤ err) (solve : ‖Un estimate‖ ≤ res)
    (small : err+res < gap) : ‖estimate-root‖ < eps := by
  by_contra h
  have hs := sep estimate (le_of_not_gt h)
  have ht := norm_sub_le (U estimate) (Un estimate)
  have he : ‖U estimate-Un estimate‖ ≤ err := by
    rw [norm_sub_rev]; exact uniform estimate
  have hb : ‖U estimate‖ ≤ err+res := calc
    ‖U estimate‖ = ‖(U estimate-Un estimate)+Un estimate‖ := by simp
    _ ≤ ‖U estimate-Un estimate‖+‖Un estimate‖ := norm_add_le _ _
    _ ≤ err+res := add_le_add he solve
  linarith

end Root

section Linear
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The sign convention is A = negative derivative and Atilde delta = s-r. -/
theorem linear_expansion (A R T : E →L[ℝ] E) (delta s r : E)
    (left : ∀ x, R (A x) = x) (segment : T delta = s-r) :
    delta-R s = R ((A-T) delta)-R r := by
  have h : (A-T) delta = A delta-(s-r) := by simp [segment]
  rw [h, map_sub, map_sub, left]
  abel

/-- A quantitative bound, obtained by absorbing the derivative perturbation.
s and r may already be multiplied by sqrt(N). -/
theorem linear_remainder_bound (A R T : E →L[ℝ] E) (delta s r : E)
    (left : ∀ x, R (A x) = x) (segment : T delta = s-r)
    (small : ‖R‖*‖A-T‖ ≤ 1/2) :
    ‖delta-R s‖ ≤ 2*‖R‖^2*‖A-T‖*(‖s‖+‖r‖)+‖R‖*‖r‖ := by
  have eq := linear_expansion A R T delta s r left segment
  have hr := R.le_opNorm r
  have hs := R.le_opNorm s
  have ht := (A-T).le_opNorm delta
  have hrt := R.le_opNorm ((A-T) delta)
  have hp : ‖R ((A-T) delta)‖ ≤ ‖R‖*‖A-T‖*‖delta‖ := by
    nlinarith [norm_nonneg R]
  have hd : ‖delta‖ ≤ ‖R‖*(‖s‖+‖r‖)+‖R‖*‖A-T‖*‖delta‖ := calc
    ‖delta‖ = ‖(delta-R s)+R s‖ := by simp
    _ ≤ ‖delta-R s‖+‖R s‖ := norm_add_le _ _
    _ ≤ (‖R ((A-T) delta)‖+‖R r‖)+‖R s‖ := by rw [eq]; gcongr; exact norm_sub_le _ _
    _ ≤ _ := by nlinarith
  have hd2 : ‖delta‖ ≤ 2*‖R‖*(‖s‖+‖r‖) := by
    nlinarith [mul_le_mul_of_nonneg_right small (norm_nonneg delta)]
  calc
    ‖delta-R s‖ ≤ ‖R ((A-T) delta)‖+‖R r‖ := by rw [eq]; exact norm_sub_le _ _
    _ ≤ ‖R‖*‖A-T‖*‖delta‖+‖R‖*‖r‖ := add_le_add hp hr
    _ ≤ _ := by nlinarith [mul_le_mul_of_nonneg_left hd2 (mul_nonneg (norm_nonneg R) (norm_nonneg (A-T)))]

/-- Genuine invertibility of nearby bread, using the Banach algebra theorem. -/
theorem nearby_bread_isUnit [CompleteSpace E] (A : (E →L[ℝ] E)ˣ)
    (T : E →L[ℝ] E) (h : ‖T-(A : E →L[ℝ] E)‖ < ‖(↑A⁻¹ : E →L[ℝ] E)‖⁻¹) :
    IsUnit T := (A.ofNearby T h).isUnit

theorem inverse_norm_bound (A R T S : E →L[ℝ] E)
    (left : ∀ x, R (A x) = x) (right : ∀ y, T (S y) = y)
    (small : ‖R‖*‖A-T‖ ≤ 1/2) : ‖S‖ ≤ 2*‖R‖ := by
  apply S.opNorm_le_bound (by positivity)
  intro y
  have he := linear_remainder_bound A R T (S y) y 0 left (by simpa using right y) small
  simp only [norm_zero, add_zero, mul_zero] at he
  have ht : ‖S y‖ ≤ ‖S y-R y‖+‖R y‖ := by
    simpa using norm_add_le (S y-R y) (R y)
  have hr := R.le_opNorm y
  have hsm := mul_le_mul_of_nonneg_left small (mul_nonneg (norm_nonneg R) (norm_nonneg y))
  nlinarith

end Linear

section Moments
variable {I D C : Type*} [Fintype I]
def avg (x : I → ℝ) : ℝ := (Fintype.card I : ℝ)⁻¹ * ∑ i, x i

theorem avg_add (x y : I → ℝ) : avg (fun i => x i+y i) = avg x+avg y := by
  unfold avg
  rw [Finset.sum_add_distrib, mul_add]
theorem avg_sub (x y : I → ℝ) : avg (fun i => x i-y i) = avg x-avg y := by
  unfold avg
  rw [Finset.sum_sub_distrib, mul_sub]
theorem avg_mul (a : ℝ) (x : I → ℝ) : avg (fun i => a*x i) = a*avg x := by
  unfold avg
  rw [← Finset.mul_sum]
  ring
theorem avg_const [Nonempty I] (a : ℝ) : avg (fun _ : I => a) = a := by
  have hn : (Fintype.card I : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  simp [avg, hn]

/-- Actual finite-row mean, compatible with the prior finiteMean. -/
theorem avg_eq_finiteMean (u : I → D → ℝ) (j : D) :
    avg (fun i => u i j) = JointSandwich.finiteMean Finset.univ u j := by
  simp [avg, JointSandwich.finiteMean, Finset.sum_apply]

def centeredMoment (u : I → D → ℝ) : Matrix D D ℝ := fun j k =>
  avg (fun i => (u i j-avg (fun l => u l j))*(u i k-avg (fun l => u l k)))

/-- This is an equality of actual finite means, not a definition of a limit. -/
theorem centeredMoment_eq [Nonempty I] (u : I → D → ℝ) (j k : D) :
    centeredMoment u j k = avg (fun i => u i j*u i k)-
      avg (fun i => u i j)*avg (fun i => u i k) := by
  unfold centeredMoment
  have he : (fun i => (u i j-avg (fun l => u l j))*(u i k-avg (fun l => u l k))) =
      (fun i => u i j*u i k - avg (fun l => u l k)*u i j -
        avg (fun l => u l j)*u i k + avg (fun l => u l j)*avg (fun l => u l k)) := by
    funext i; ring
  rw [he, avg_add, avg_sub, avg_sub, avg_mul, avg_mul, avg_const]
  ring

/-- Cauchy--Schwarz for empirical averages, with no bounded-row assumption. -/
theorem avg_product_sq_le (x y : I → ℝ) :
    (avg (fun i => x i*y i))^2 ≤ avg (fun i => (x i)^2)*avg (fun i => (y i)^2) := by
  have h := Finset.sum_mul_sq_le_sq_mul_sq (s := Finset.univ) x y
  unfold avg
  nlinarith [mul_le_mul_of_nonneg_left h (sq_nonneg ((Fintype.card I : ℝ)⁻¹))]

theorem mean_perturbation_sq [Nonempty I] (x y : I → ℝ) :
    (avg x-avg y)^2 ≤ avg (fun i => (x i-y i)^2) := by
  have h := avg_product_sq_le (fun i => x i-y i) (fun _ => (1:ℝ))
  simpa [avg_sub, avg_const] using h

/-- Second moments use two reference second moments and the L2 score perturbations. -/
theorem second_perturbation_sq (x y dx dy : I → ℝ) :
    (avg (fun i => (x i+dx i)*(y i+dy i))-avg (fun i => x i*y i))^2 ≤
      3*(avg (fun i => (x i)^2)*avg (fun i => (dy i)^2)+
        avg (fun i => (dx i)^2)*avg (fun i => (y i)^2)+
        avg (fun i => (dx i)^2)*avg (fun i => (dy i)^2)) := by
  have h1 := avg_product_sq_le x dy
  have h2 := avg_product_sq_le dx y
  have h3 := avg_product_sq_le dx dy
  have he : (fun i => (x i+dx i)*(y i+dy i)) =
      (fun i => (x i*y i+x i*dy i)+(dx i*y i+dx i*dy i)) := by funext i; ring
  rw [he, avg_add, avg_add, avg_add]
  nlinarith [sq_nonneg (avg (fun i => x i*dy i)-avg (fun i => dx i*y i)),
    sq_nonneg (avg (fun i => x i*dy i)-avg (fun i => dx i*dy i)),
    sq_nonneg (avg (fun i => dx i*y i)-avg (fun i => dx i*dy i))]

/-- Lipschitz envelope supplies the requested average-square fit perturbation. -/
theorem lipschitz_square_bound (x y L : I → ℝ) (distance : ℝ)
    (h : ∀ i, |x i-y i| ≤ L i*distance) :
    avg (fun i => (x i-y i)^2) ≤ avg (fun i => (L i)^2)*distance^2 := by
  unfold avg
  have hs : (∑ i, (x i-y i)^2) ≤ ∑ i, (L i)^2*distance^2 := by
    apply Finset.sum_le_sum
    intro i hi
    have hh := h i
    have hab := abs_nonneg (x i-y i)
    nlinarith [sq_abs (x i-y i)]
  have hm := mul_le_mul_of_nonneg_left hs (inv_nonneg.mpr (Nat.cast_nonneg (Fintype.card I)) : (0:ℝ) ≤ (Fintype.card I : ℝ)⁻¹)
  simpa [Finset.sum_mul, mul_assoc] using hm

end Moments

section Limits
variable {J C D : Type*} [Fintype C] [Fintype D] [DecidableEq D]
variable {l : Filter J}

omit [Fintype D] [DecidableEq D] in
/-- Bread at the fitted root is a reference derivative mean plus local derivative perturbation. -/
theorem bread_limit (ref perturb : J → Matrix D D ℝ) (A : Matrix D D ℝ)
    (h : Tendsto ref l (𝓝 A)) (hp : Tendsto perturb l (𝓝 0)) :
    Tendsto (fun n => ref n+perturb n) l (𝓝 A) := by simpa using h.add hp

def meatFromMoments (w : C → ℝ) (m : C → D → ℝ)
    (Q : C → Matrix D D ℝ) : Matrix D D ℝ :=
  fun j k => ∑ c, w c*(Q c j k-m c j*m c k)

omit [Fintype D] [DecidableEq D] in
/-- Finite-center/coordinate continuous combination of raw moments and weights. -/
theorem meat_limit (w : J → C → ℝ) (m : J → C → D → ℝ)
    (Q : J → C → Matrix D D ℝ) (pi : C → ℝ) (m0 : C → D → ℝ)
    (Q0 : C → Matrix D D ℝ)
    (hw : ∀ c, Tendsto (fun n => w n c) l (𝓝 (pi c)))
    (hm : ∀ c j, Tendsto (fun n => m n c j) l (𝓝 (m0 c j)))
    (hq : ∀ c j k, Tendsto (fun n => Q n c j k) l (𝓝 (Q0 c j k))) :
    Tendsto (fun n => meatFromMoments (w n) (m n) (Q n)) l
      (𝓝 (meatFromMoments pi m0 Q0)) := by
  apply tendsto_pi_nhds.mpr; intro j
  apply tendsto_pi_nhds.mpr; intro k
  exact tendsto_finsetSum _ (fun c hc => (hw c).mul ((hq c j k).sub ((hm c j).mul (hm c k))))

/-- Nonsingularity at the limit, not consistency of an assumed inverse estimator. -/
theorem inverse_limit (A : J → Matrix D D ℝ) (A0 : Matrix D D ℝ)
    (h : Tendsto A l (𝓝 A0)) (hu : IsUnit A0.det) :
    Tendsto (fun n => (A n)⁻¹) l (𝓝 A0⁻¹) :=
  (continuousAt_matrix_inv A0 (by
    obtain ⟨u, hu⟩ := hu
    rw [← hu]
    exact NormedRing.inverse_continuousAt u)).tendsto.comp h

def sandwichScalar (h : D → ℝ) (R B : Matrix D D ℝ) : ℝ :=
  h ⬝ᵥ ((R*B*R.transpose)*ᵥ h)

omit [DecidableEq D] in
theorem sandwich_limit (h : J → D → ℝ) (R B : J → Matrix D D ℝ)
    (h0 : D → ℝ) (R0 B0 : Matrix D D ℝ)
    (hh : Tendsto h l (𝓝 h0)) (hr : Tendsto R l (𝓝 R0))
    (hb : Tendsto B l (𝓝 B0)) :
    Tendsto (fun n => sandwichScalar (h n) (R n) (B n)) l
      (𝓝 (sandwichScalar h0 R0 B0)) := by
  have hc : Continuous (fun z : (D → ℝ) × Matrix D D ℝ × Matrix D D ℝ =>
      sandwichScalar z.1 z.2.1 z.2.2) := by
    simp only [sandwichScalar, Matrix.mulVec, dotProduct, Matrix.mul_apply, Matrix.transpose_apply]
    fun_prop
  exact hc.tendsto _ |>.comp (hh.prodMk_nhds (hr.prodMk_nhds hb))

end Limits

section ProbabilityMechanisms
variable {Ω E F : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
  [PseudoMetricSpace E] [PseudoMetricSpace F]

/-- Continuous mapping at a constant, proved by a direct bad-event inclusion. -/
theorem inMeasure_continuousAt (X : ℕ → Ω → E) (x : E) (g : E → F)
    (hx : TendstoInMeasure μ X atTop (fun _ => x)) (hg : ContinuousAt g x) :
    TendstoInMeasure μ (fun n ω => g (X n ω)) atTop (fun _ => g x) := by
  rw [tendstoInMeasure_iff_dist] at hx ⊢
  intro eps heps
  obtain ⟨d, hd, hgd⟩ := Metric.continuousAt_iff.mp hg eps heps
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds (hx d hd)
  · intro n; exact bot_le
  · intro n
    apply measure_mono
    intro ω hω
    exact le_of_not_gt (fun h => (not_le_of_gt (hgd h)) hω)

theorem inMeasure_pair (X : ℕ → Ω → E) (Y : ℕ → Ω → F) (x : E) (y : F)
    (hx : TendstoInMeasure μ X atTop (fun _ => x))
    (hy : TendstoInMeasure μ Y atTop (fun _ => y)) :
    TendstoInMeasure μ (fun n ω => (X n ω,Y n ω)) atTop (fun _ => (x,y)) := by
  rw [tendstoInMeasure_iff_dist] at hx hy ⊢
  intro eps heps
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
    (by simpa using (hx eps heps).add (hy eps heps))
  · intro n; exact bot_le
  · intro n
    refine (measure_mono (fun ω hω => ?_)).trans (measure_union_le _ _)
    simpa only [Set.mem_setOf_eq, Set.mem_union, Prod.dist_eq, le_max_iff] using hω

theorem inMeasure_finite {D G : Type*} [Fintype D] [NormedAddCommGroup G]
    (X : ℕ → Ω → D → G) (x : D → G)
    (hx : ∀ j, TendstoInMeasure μ (fun n ω => X n ω j) atTop (fun _ => x j)) :
    TendstoInMeasure μ X atTop (fun _ => x) := by
  rw [tendstoInMeasure_iff_norm]
  intro eps heps
  have hsum : Tendsto (fun n => ∑ j, μ {ω | eps ≤ ‖X n ω j-x j‖}) atTop (𝓝 0) := by
    simpa using tendsto_finsetSum Finset.univ (fun j _ => (tendstoInMeasure_iff_norm.mp (hx j)) eps heps)
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hsum
  · intro n; exact bot_le
  · intro n
    refine (measure_mono (show {ω | eps ≤ ‖X n ω-x‖} ⊆ ⋃ j, {ω | eps ≤ ‖X n ω j-x j‖} from ?_)).trans (measure_iUnion_fintype_le μ _)
    intro ω hω
    simp only [Set.mem_iUnion, Set.mem_setOf_eq]
    by_contra hn
    push Not at hn
    exact (not_le_of_gt ((pi_norm_lt_iff heps).mpr hn)) hω

/-- Square domination turns empirical L2 bounds into o_p(1) directly. -/
theorem inMeasure_of_sq_le (X B : ℕ → Ω → ℝ)
    (h : ∀ n ω, (X n ω)^2 ≤ B n ω)
    (hb : TendstoInMeasure μ B atTop 0) : TendstoInMeasure μ X atTop 0 := by
  rw [tendstoInMeasure_iff_norm] at hb ⊢
  intro eps heps
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds (hb (eps^2) (sq_pos_of_pos heps))
  · intro n; exact bot_le
  · intro n
    apply measure_mono
    intro ω hω
    simp only [Set.mem_setOf_eq, Pi.zero_apply, sub_zero, Real.norm_eq_abs] at hω ⊢
    have hsq := sq_abs (X n ω)
    have hh := h n ω
    have hba := le_abs_self (B n ω)
    nlinarith

theorem inMeasure_add (X Y : ℕ → Ω → ℝ) (x y : ℝ)
    (hx : TendstoInMeasure μ X atTop (fun _ => x))
    (hy : TendstoInMeasure μ Y atTop (fun _ => y)) :
    TendstoInMeasure μ (fun n ω => X n ω+Y n ω) atTop (fun _ => x+y) := by
  have hp := inMeasure_pair μ X Y x y hx hy
  have hc : ContinuousAt (fun p : ℝ×ℝ => p.1+p.2) (x,y) := (continuous_fst.add continuous_snd).continuousAt
  exact inMeasure_continuousAt μ _ (x,y) _ hp hc

theorem inMeasure_mul (X Y : ℕ → Ω → ℝ) (x y : ℝ)
    (hx : TendstoInMeasure μ X atTop (fun _ => x))
    (hy : TendstoInMeasure μ Y atTop (fun _ => y)) :
    TendstoInMeasure μ (fun n ω => X n ω*Y n ω) atTop (fun _ => x*y) := by
  have hp := inMeasure_pair μ X Y x y hx hy
  have hc : ContinuousAt (fun p : ℝ×ℝ => p.1*p.2) (x,y) := (continuous_fst.mul continuous_snd).continuousAt
  exact inMeasure_continuousAt μ _ (x,y) _ hp hc

theorem inMeasure_sum {D : Type*} [Fintype D] (X : ℕ → Ω → D → ℝ) (x : D → ℝ)
    (hx : ∀ j, TendstoInMeasure μ (fun n ω => X n ω j) atTop (fun _ => x j)) :
    TendstoInMeasure μ (fun n ω => ∑ j, X n ω j) atTop (fun _ => ∑ j, x j) := by
  have hp := inMeasure_finite μ X x hx
  have hc : Continuous (fun z : D → ℝ => ∑ j, z j) := by fun_prop
  exact inMeasure_continuousAt μ X x (fun z : D → ℝ => ∑ j, z j) hp hc.continuousAt

/-- Sample sizes may vary with n; the reference rows and fitted rows are actual arrays. -/
theorem fitted_mean_inMeasure {I : ℕ → Type*} [∀ n, Fintype (I n)] [∀ n, Nonempty (I n)]
    (u v : (n : ℕ) → Ω → I n → ℝ) (m : ℝ)
    (hu : TendstoInMeasure μ (fun n ω => avg (u n ω)) atTop (fun _ => m))
    (hd : TendstoInMeasure μ (fun n ω => avg (fun i => (v n ω i-u n ω i)^2)) atTop 0) :
    TendstoInMeasure μ (fun n ω => avg (v n ω)) atTop (fun _ => m) := by
  have he := inMeasure_of_sq_le μ _ _ (fun n ω => mean_perturbation_sq (v n ω) (u n ω)) hd
  have hs := inMeasure_add μ _ _ m 0 hu he
  simpa using hs

theorem fitted_second_inMeasure {I : ℕ → Type*} [∀ n, Fintype (I n)]
    (x y dx dy : (n : ℕ) → Ω → I n → ℝ) (qx qy qxy : ℝ)
    (hx : TendstoInMeasure μ (fun n ω => avg (fun i => (x n ω i)^2)) atTop (fun _ => qx))
    (hy : TendstoInMeasure μ (fun n ω => avg (fun i => (y n ω i)^2)) atTop (fun _ => qy))
    (hxy : TendstoInMeasure μ (fun n ω => avg (fun i => x n ω i*y n ω i)) atTop (fun _ => qxy))
    (hdx : TendstoInMeasure μ (fun n ω => avg (fun i => (dx n ω i)^2)) atTop 0)
    (hdy : TendstoInMeasure μ (fun n ω => avg (fun i => (dy n ω i)^2)) atTop 0) :
    TendstoInMeasure μ (fun n ω => avg (fun i => (x n ω i+dx n ω i)*(y n ω i+dy n ω i))) atTop (fun _ => qxy) := by
  have hb1 := inMeasure_mul μ _ _ qx 0 hx hdy
  have hb2 := inMeasure_mul μ _ _ 0 qy hdx hy
  have hb3 := inMeasure_mul μ _ _ 0 0 hdx hdy
  have hb := inMeasure_add μ _ _ _ _ (inMeasure_add μ _ _ _ _ hb1 hb2) hb3
  have hb' := inMeasure_continuousAt μ _ _ (fun z : ℝ => 3*z) hb (by fun_prop)
  have he := inMeasure_of_sq_le μ _ _ (fun n ω => second_perturbation_sq (x n ω) (y n ω) (dx n ω) (dy n ω)) (by simpa only [mul_zero, zero_mul, add_zero, Pi.zero_def, add_assoc] using hb')
  have hs := inMeasure_add μ _ _ qxy 0 hxy he
  simpa using hs

/-- A good-event deterministic envelope produces the M-remainder probability input. -/
theorem inMeasure_of_good_envelope (X : ℕ → Ω → ℝ) (b : ℕ → ℝ) (good : ℕ → Set Ω)
    (bound : ∀ n ω, ω ∈ good n → |X n ω| ≤ b n)
    (hb : Tendsto b atTop (𝓝 0)) (hg : Tendsto (fun n => μ (good n)ᶜ) atTop (𝓝 0)) :
    TendstoInMeasure μ X atTop 0 := by
  rw [tendstoInMeasure_iff_norm]
  intro eps heps
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hg
  · exact Filter.Eventually.of_forall (fun n => bot_le)
  · filter_upwards [hb.eventually (gt_mem_nhds heps)] with n hn
    apply measure_mono
    intro ω hω
    change ω ∉ good n
    intro hw
    have hh := bound n ω hw
    simp only [Set.mem_setOf_eq, Pi.zero_apply, sub_zero, Real.norm_eq_abs] at hω
    linarith

end ProbabilityMechanisms

section RootProbability
variable {Ω E F : Type*} [MeasurableSpace Ω] [NormedAddCommGroup E]
  [NormedAddCommGroup F] (μ : Measure Ω)

/-- Compact parameter domains are represented by S. The ULLN itself is external;
this proves its probability composition with separation and numerical residuals. -/
theorem root_consistency_inMeasure (S : Set E) (U : ℕ → E → F)
    (Un : ℕ → Ω → E → F) (root : ℕ → E) (estimate : ℕ → Ω → E)
    (err res : ℕ → Ω → ℝ) (hm : ∀ n ω, estimate n ω ∈ S)
    (sep : ∀ eps, 0 < eps → ∃ gap, 0 < gap ∧ ∀ᶠ n in atTop,
      ∀ x ∈ S, eps ≤ ‖x-root n‖ → gap ≤ ‖U n x‖)
    (uniform : ∀ n ω x, x ∈ S → ‖Un n ω x-U n x‖ ≤ err n ω)
    (solve : ∀ n ω, ‖Un n ω (estimate n ω)‖ ≤ res n ω)
    (he : TendstoInMeasure μ err atTop 0) (hr : TendstoInMeasure μ res atTop 0) :
    TendstoInMeasure μ (fun n ω => estimate n ω-root n) atTop 0 := by
  have hs := inMeasure_add μ err res 0 0 he hr
  rw [tendstoInMeasure_iff_norm] at hs ⊢
  intro eps heps
  obtain ⟨gap, hgap, hn⟩ := sep eps heps
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds (hs gap hgap)
  · exact Filter.Eventually.of_forall (fun n => bot_le)
  · filter_upwards [hn] with n hn
    apply measure_mono
    intro ω hω
    simp only [Set.mem_setOf_eq, Pi.zero_apply, sub_zero] at hω
    have hg := hn (estimate n ω) (hm n ω) hω
    have hu := uniform n ω (estimate n ω) (hm n ω)
    have hv := solve n ω
    have hb : ‖U n (estimate n ω)‖ ≤ err n ω+res n ω := calc
      _ = ‖(U n (estimate n ω)-Un n ω (estimate n ω))+Un n ω (estimate n ω)‖ := by simp
      _ ≤ ‖U n (estimate n ω)-Un n ω (estimate n ω)‖+‖Un n ω (estimate n ω)‖ := norm_add_le _ _
      _ ≤ _ := by rw [norm_sub_rev]; exact add_le_add hu hv
    simp only [Set.mem_setOf_eq, add_zero, sub_zero, Real.norm_eq_abs]
    exact hg.trans (hb.trans (le_abs_self _))

end RootProbability

section ActualVariance
variable {Ω C D : Type*} [MeasurableSpace Ω] [Fintype C] [Fintype D] [DecidableEq D]
  (μ : Measure Ω)

def actualMeat {I : C → Type*} [∀ c, Fintype (I c)]
    (w : C → ℝ) (u : ∀ c, I c → D → ℝ) : Matrix D D ℝ :=
  fun j k => ∑ c, w c*centeredMoment (u c) j k

omit [Fintype D] [DecidableEq D] in
theorem actualMeat_eq {I : C → Type*} [∀ c, Fintype (I c)] [∀ c, Nonempty (I c)]
    (w : C → ℝ) (u : ∀ c, I c → D → ℝ) :
    actualMeat w u = meatFromMoments w (fun c j => avg (fun i => u c i j))
      (fun c j k => avg (fun i => u c i j*u c i k)) := by
  ext j k
  simp only [actualMeat, meatFromMoments, centeredMoment_eq]

omit [DecidableEq D] in
/-- The primitive inputs are reference first/second-moment LLNs and fitted-score
mean-square perturbations, for each finite center/coordinate. -/
theorem actualMeat_inMeasure {I : ℕ → C → Type*}
    [∀ n c, Fintype (I n c)] [∀ n c, Nonempty (I n c)]
    (w : ℕ → Ω → C → ℝ) (u v : (n : ℕ) → Ω → (c : C) → I n c → D → ℝ)
    (pi : C → ℝ) (m : C → D → ℝ) (Q : C → Matrix D D ℝ)
    (hw : ∀ c, TendstoInMeasure μ (fun n ω => w n ω c) atTop (fun _ => pi c))
    (hm : ∀ c j, TendstoInMeasure μ (fun n ω => avg (fun i => u n ω c i j)) atTop (fun _ => m c j))
    (hq : ∀ c j k, TendstoInMeasure μ (fun n ω => avg (fun i => u n ω c i j*u n ω c i k)) atTop (fun _ => Q c j k))
    (hd : ∀ c j, TendstoInMeasure μ (fun n ω => avg (fun i => (v n ω c i j-u n ω c i j)^2)) atTop 0) :
    TendstoInMeasure μ (fun n ω => actualMeat (w n ω) (v n ω)) atTop
      (fun _ => meatFromMoments pi m Q) := by
  have hm' : ∀ c j, TendstoInMeasure μ (fun n ω => avg (fun i => v n ω c i j)) atTop (fun _ => m c j) := by
    intro c j
    exact fitted_mean_inMeasure μ _ _ _ (hm c j) (hd c j)
  have hq' : ∀ c j k, TendstoInMeasure μ (fun n ω => avg (fun i => v n ω c i j*v n ω c i k)) atTop (fun _ => Q c j k) := by
    intro c j k
    have ht := fitted_second_inMeasure μ (fun n ω i => u n ω c i j) (fun n ω i => u n ω c i k)
      (fun n ω i => v n ω c i j-u n ω c i j) (fun n ω i => v n ω c i k-u n ω c i k)
      (Q c j j) (Q c k k) (Q c j k) (by simpa [pow_two] using hq c j j)
      (by simpa [pow_two] using hq c k k) (hq c j k) (hd c j) (hd c k)
    simpa using ht
  have hentry : ∀ j k, TendstoInMeasure μ (fun n ω => actualMeat (w n ω) (v n ω) j k)
      atTop (fun _ => meatFromMoments pi m Q j k) := by
    intro j k
    have ht : ∀ c, TendstoInMeasure μ (fun n ω => w n ω c *
        (avg (fun i => v n ω c i j*v n ω c i k)-
          avg (fun i => v n ω c i j)*avg (fun i => v n ω c i k)))
        atTop (fun _ => pi c*(Q c j k-m c j*m c k)) := by
      intro c
      have hp := inMeasure_mul μ _ _ _ _ (hm' c j) (hm' c k)
      have hpair := inMeasure_pair μ _ _ _ _ (hq' c j k) hp
      have hc : ContinuousAt (fun z : ℝ×ℝ => z.1-z.2) (Q c j k,m c j*m c k) := by fun_prop
      have hsub := inMeasure_continuousAt μ _ _ (fun z : ℝ×ℝ => z.1-z.2) hpair hc
      exact inMeasure_mul μ _ _ _ _ (hw c) hsub
    have hv := inMeasure_finite μ _ _ ht
    have hc : Continuous (fun z : C → ℝ => ∑ c, z c) := by fun_prop
    have hs := inMeasure_continuousAt μ _ _ (fun z : C → ℝ => ∑ c, z c) hv hc.continuousAt
    simpa only [actualMeat, meatFromMoments, centeredMoment_eq] using hs
  exact inMeasure_finite μ _ _ (fun j => inMeasure_finite μ _ _ (hentry j))

/-- Actual bread = reference derivative average + fitted derivative perturbation.
The scalar is the N-times variance; h is the person-weight extraction row. -/
theorem actual_sandwich_inMeasure
    (h : ℕ → Ω → D → ℝ) (Aref Apert B : ℕ → Ω → Matrix D D ℝ)
    (h0 : D → ℝ) (A0 B0 : Matrix D D ℝ) (hu : IsUnit A0.det)
    (hh : TendstoInMeasure μ h atTop (fun _ => h0))
    (ha : TendstoInMeasure μ Aref atTop (fun _ => A0))
    (hd : TendstoInMeasure μ Apert atTop 0)
    (hb : TendstoInMeasure μ B atTop (fun _ => B0)) :
    TendstoInMeasure μ (fun n ω => sandwichScalar (h n ω) ((Aref n ω+Apert n ω)⁻¹) (B n ω))
      atTop (fun _ => sandwichScalar h0 A0⁻¹ B0) := by
  have hpair := inMeasure_pair μ Aref Apert A0 0 ha hd
  have hcadd : ContinuousAt (fun z : Matrix D D ℝ × Matrix D D ℝ => z.1+z.2) (A0,0) := by fun_prop
  have ha' := inMeasure_continuousAt μ _ (A0,0) (fun z : Matrix D D ℝ × Matrix D D ℝ => z.1+z.2) hpair hcadd
  have hinv : ContinuousAt (fun A : Matrix D D ℝ => A⁻¹) A0 := by
    apply continuousAt_matrix_inv A0
    obtain ⟨u, he⟩ := hu
    rw [← he]
    exact NormedRing.inverse_continuousAt u
  have hr := inMeasure_continuousAt μ _ A0 (fun A : Matrix D D ℝ => A⁻¹) (by simpa using ha') hinv
  have hhc : ∀ i, TendstoInMeasure μ (fun n ω => h n ω i) atTop (fun _ => h0 i) := by
    intro i
    exact inMeasure_continuousAt μ h h0 (fun z => z i) hh (continuous_apply i).continuousAt
  have hrc : ∀ i j, TendstoInMeasure μ (fun n ω => ((Aref n ω+Apert n ω)⁻¹) i j) atTop (fun _ => A0⁻¹ i j) := by
    intro i j
    exact inMeasure_continuousAt μ _ A0⁻¹ (fun z : Matrix D D ℝ => z i j) hr
      ((continuous_apply j).comp (continuous_apply i)).continuousAt
  have hbc : ∀ i j, TendstoInMeasure μ (fun n ω => B n ω i j) atTop (fun _ => B0 i j) := by
    intro i j
    exact inMeasure_continuousAt μ B B0 (fun z => z i j) hb
      ((continuous_apply j).comp (continuous_apply i)).continuousAt
  simp only [sandwichScalar, Matrix.mulVec, dotProduct, Matrix.mul_apply, Matrix.transpose_apply]
  apply inMeasure_sum μ
  intro i
  apply inMeasure_mul μ _ _ _ _ (hhc i)
  apply inMeasure_sum μ
  intro j
  apply inMeasure_mul μ _ _ _ _ _ (hhc j)
  apply inMeasure_sum μ
  intro k
  apply inMeasure_mul μ _ _ _ _ _ (hrc j k)
  apply inMeasure_sum μ
  intro t
  exact inMeasure_mul μ _ _ _ _ (hrc i t) (hbc t k)

end ActualVariance

section RemainderProbability
variable {Ω E : Type*} [MeasurableSpace Ω] [NormedAddCommGroup E] [NormedSpace ℝ E]
  (μ : Measure Ω)

/-- Primitive high-probability derivative, row-sum, and solver-residual envelopes.
No final remainder convergence is assumed. Rates permit growing score envelopes. -/
theorem linear_remainder_inMeasure (A R : E →L[ℝ] E)
    (T : ℕ → Ω → E →L[ℝ] E) (delta s r : ℕ → Ω → E)
    (d z q : ℕ → ℝ) (good : ℕ → Set Ω)
    (left : ∀ x, R (A x) = x)
    (segment : ∀ n ω, ω ∈ good n → T n ω (delta n ω) = s n ω-r n ω)
    (bounds : ∀ n ω, ω ∈ good n → ‖A-T n ω‖ ≤ d n ∧ ‖s n ω‖ ≤ z n ∧ ‖r n ω‖ ≤ q n)
    (nonneg : ∀ n, 0 ≤ d n ∧ 0 ≤ z n ∧ 0 ≤ q n)
    (small : ∀ n, ‖R‖*d n ≤ 1/2)
    (hdz : Tendsto (fun n => d n*z n) atTop (𝓝 0))
    (hdq : Tendsto (fun n => d n*q n) atTop (𝓝 0))
    (hq : Tendsto q atTop (𝓝 0))
    (hg : Tendsto (fun n => μ (good n)ᶜ) atTop (𝓝 0)) :
    TendstoInMeasure μ (fun n ω => ‖delta n ω-R (s n ω)‖) atTop 0 := by
  apply inMeasure_of_good_envelope μ _ (fun n => 2*‖R‖^2*(d n*z n+d n*q n)+‖R‖*q n) good
  · intro n ω hw
    obtain ⟨ht, hs, hr⟩ := bounds n ω hw
    have hh := linear_remainder_bound A R (T n ω) (delta n ω) (s n ω) (r n ω) left (segment n ω hw)
      (le_trans (mul_le_mul_of_nonneg_left ht (norm_nonneg R)) (small n))
    rw [abs_of_nonneg (norm_nonneg _)]
    calc
      _ ≤ _ := hh
      _ ≤ 2*‖R‖^2*d n*(z n+q n)+‖R‖*q n := by
        have hd := (nonneg n).1
        gcongr
      _ = _ := by ring
  · simpa using ((tendsto_const_nhds.mul (hdz.add hdq)).add (tendsto_const_nhds.mul hq))
  · exact hg

end RemainderProbability

#print axioms root_localization
#print axioms root_consistency_inMeasure
#print axioms linear_expansion
#print axioms linear_remainder_bound
#print axioms nearby_bread_isUnit
#print axioms inverse_norm_bound
#print axioms avg_add
#print axioms avg_sub
#print axioms avg_mul
#print axioms avg_const
#print axioms avg_eq_finiteMean
#print axioms centeredMoment_eq
#print axioms avg_product_sq_le
#print axioms mean_perturbation_sq
#print axioms second_perturbation_sq
#print axioms lipschitz_square_bound
#print axioms bread_limit
#print axioms meat_limit
#print axioms inverse_limit
#print axioms sandwich_limit
#print axioms inMeasure_continuousAt
#print axioms inMeasure_pair
#print axioms inMeasure_finite
#print axioms inMeasure_of_sq_le
#print axioms inMeasure_add
#print axioms inMeasure_mul
#print axioms inMeasure_sum
#print axioms fitted_mean_inMeasure
#print axioms fitted_second_inMeasure
#print axioms inMeasure_of_good_envelope
#print axioms actualMeat_eq
#print axioms actualMeat_inMeasure
#print axioms actual_sandwich_inMeasure
#print axioms linear_remainder_inMeasure

end FixedCenterM
