import Mathlib.Data.Matrix.Block
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring

/-! Finite, deterministic joint-equation sandwich algebra.
No claim about numerical roots, derivatives, CLTs, or interval calibration. -/
noncomputable section
open Matrix
open scoped BigOperators
namespace JointSandwich

variable {p d r t : Type*}

def bread [DecidableEq p] (J : Matrix p p ℝ) (c : Matrix Unit p ℝ) :
    Matrix (p ⊕ Unit) (p ⊕ Unit) ℝ :=
  Matrix.fromBlocks J 0 (-c) 1

def inverseCandidate [Fintype p] (K : Matrix p p ℝ) (c : Matrix Unit p ℝ) :
    Matrix (p ⊕ Unit) (p ⊕ Unit) ℝ :=
  Matrix.fromBlocks K 0 (c * K) 1

def meat [Fintype t] (N : ℝ) (u : t → d → ℝ) : Matrix d d ℝ :=
  N⁻¹ • ∑ i, Matrix.vecMulVec (u i) (u i)

def covariance [Fintype d] [Fintype t] (N : ℝ) (R : Matrix d d ℝ)
    (u : t → d → ℝ) : Matrix d d ℝ :=
  N⁻¹ • (R * meat N u * R.transpose)

def finiteMean (s : Finset t) (u : t → d → ℝ) : d → ℝ :=
  (s.card : ℝ)⁻¹ • ∑ i ∈ s, u i

theorem bread_mul_inverseCandidate [Fintype p] [DecidableEq p]
    (J K : Matrix p p ℝ) (c : Matrix Unit p ℝ) (hJK : J * K = 1) :
    bread J c * inverseCandidate K c = 1 := by
  simp [bread, inverseCandidate, Matrix.fromBlocks_multiply, hJK]

theorem inverseCandidate_mul_bread [Fintype p] [DecidableEq p]
    (J K : Matrix p p ℝ) (c : Matrix Unit p ℝ) (hKJ : K * J = 1) :
    inverseCandidate K c * bread J c = 1 := by
  simp [bread, inverseCandidate, Matrix.fromBlocks_multiply, Matrix.mul_assoc, hKJ]

theorem last_coordinate [Fintype p] (K : Matrix p p ℝ) (c : Matrix Unit p ℝ)
    (g : p → ℝ) (q theta : ℝ) :
    (inverseCandidate K c *ᵥ Sum.elim g (fun _ => q - theta)) (Sum.inr ()) =
      q - theta + c () ⬝ᵥ (K *ᵥ g) := by
  unfold inverseCandidate
  rw [Matrix.fromBlocks_mulVec]
  simp only [Function.comp_def, Sum.elim_inl, Sum.elim_inr, Matrix.one_mulVec,
    Pi.add_apply, ← Matrix.mulVec_mulVec]
  exact add_comm _ _

theorem transform_outer [Fintype d] (R : Matrix r d ℝ) (u : d → ℝ) :
    R * Matrix.vecMulVec u u * R.transpose =
      Matrix.vecMulVec (R *ᵥ u) (R *ᵥ u) := by
  rw [Matrix.mul_vecMulVec, Matrix.vecMulVec_mul, Matrix.vecMul_transpose]

theorem outer_quadratic [Fintype d] (v z : d → ℝ) :
    z ⬝ᵥ (Matrix.vecMulVec v v *ᵥ z) = (z ⬝ᵥ v) ^ 2 := by
  rw [Matrix.dotProduct_mulVec, Matrix.vecMul_vecMulVec, smul_dotProduct,
    dotProduct_comm v z]
  simp [pow_two]

/-- Full dense covariance identity; no diagonal or independence hypothesis. -/
theorem sandwich_quadratic [Fintype d] [Fintype r] [Fintype t]
    (N : ℝ) (_hN : 0 < N) (R : Matrix d d ℝ) (H : Matrix r d ℝ)
    (a : r → ℝ) (u : t → d → ℝ) :
    a ⬝ᵥ ((H * covariance N R u * H.transpose) *ᵥ a) =
      (N ^ 2)⁻¹ * ∑ i, (a ⬝ᵥ (H *ᵥ (R *ᵥ u i))) ^ 2 := by
  unfold covariance meat
  simp only [Matrix.mul_smul, Matrix.smul_mul, smul_smul,
    Matrix.mul_sum, Matrix.sum_mul]
  simp only [Matrix.smul_mulVec, Matrix.sum_mulVec, dotProduct_smul,
    dotProduct_sum, smul_eq_mul]
  rw [← pow_two, ← inv_pow]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  rw [transform_outer, transform_outer, outer_quadratic]

theorem transform_finiteMean [Fintype d] (L : Matrix r d ℝ)
    (s : Finset t) (u : t → d → ℝ) :
    L *ᵥ finiteMean s u = finiteMean s (fun i => L *ᵥ u i) := by
  simp [finiteMean, Matrix.mulVec_smul, Matrix.mulVec_sum]

/-- One fixed transform per center; the finite set is that center's members. -/
theorem transform_centered [Fintype d] (L : Matrix r d ℝ)
    (s : Finset t) (u : t → d → ℝ) (i : t) :
    L *ᵥ (u i - finiteMean s u) =
      L *ᵥ u i - finiteMean s (fun j => L *ᵥ u j) := by
  rw [Matrix.mulVec_sub, transform_finiteMean]

theorem scalar_transform_centered [Fintype d] [Fintype r]
    (R : Matrix d d ℝ) (H : Matrix r d ℝ) (a : r → ℝ)
    (s : Finset t) (u : t → d → ℝ) (i : t) :
    a ⬝ᵥ (H *ᵥ (R *ᵥ (u i - finiteMean s u))) =
      a ⬝ᵥ (H *ᵥ (R *ᵥ u i)) -
        (s.card : ℝ)⁻¹ * ∑ j ∈ s, a ⬝ᵥ (H *ᵥ (R *ᵥ u j)) := by
  rw [transform_centered, Matrix.mulVec_sub, transform_finiteMean, dotProduct_sub]
  simp [finiteMean, dotProduct_smul, dotProduct_sum]

/-- Apply the dense identity to actual centered rows, deriving the centered scalar
contribution inside each square rather than assuming that replacement. -/
theorem centered_sandwich_quadratic [Fintype d] [Fintype r] [Fintype t]
    (N : ℝ) (hN : 0 < N) (R : Matrix d d ℝ) (H : Matrix r d ℝ)
    (a : r → ℝ) (u : t → d → ℝ) (centerMembers : t → Finset t) :
    a ⬝ᵥ ((H * covariance N R (fun i => u i - finiteMean (centerMembers i) u) *
      H.transpose) *ᵥ a) =
      (N ^ 2)⁻¹ * ∑ i, (a ⬝ᵥ (H *ᵥ (R *ᵥ u i)) -
        ((centerMembers i).card : ℝ)⁻¹ *
          ∑ j ∈ centerMembers i, a ⬝ᵥ (H *ᵥ (R *ᵥ u j))) ^ 2 := by
  rw [sandwich_quadratic N hN]
  simp_rw [scalar_transform_centered]

section GlobalBlocks
variable {k : Type*} [Fintype k] [DecidableEq k] [Fintype d] [DecidableEq d]

/-- Equal-dimensional local blocks; coordinates are `(local coordinate, cluster)`. -/
def globalBread (a : k → ℝ) (A : k → Matrix d d ℝ) : Matrix (d × k) (d × k) ℝ :=
  Matrix.blockDiagonal (fun j => a j • A j)

def globalInverse (a : k → ℝ) (K : k → Matrix d d ℝ) : Matrix (d × k) (d × k) ℝ :=
  Matrix.blockDiagonal (fun j => (a j)⁻¹ • K j)

theorem global_inverse_right (a : k → ℝ) (ha : ∀ j, a j ≠ 0)
    (A K : k → Matrix d d ℝ) (hAK : ∀ j, A j * K j = 1) :
    globalBread a A * globalInverse a K = 1 := by
  unfold globalBread globalInverse
  rw [← Matrix.blockDiagonal_mul]
  have h : (fun j => (a j • A j) * ((a j)⁻¹ • K j)) = (1 : k → Matrix d d ℝ) := by
    funext j
    simp [smul_smul, hAK, ha]
  rw [h, Matrix.blockDiagonal_one]

theorem global_inverse_left (a : k → ℝ) (ha : ∀ j, a j ≠ 0)
    (A K : k → Matrix d d ℝ) (hKA : ∀ j, K j * A j = 1) :
    globalInverse a K * globalBread a A = 1 := by
  unfold globalBread globalInverse
  rw [← Matrix.blockDiagonal_mul]
  have h : (fun j => ((a j)⁻¹ • K j) * (a j • A j)) = (1 : k → Matrix d d ℝ) := by
    funext j
    simp [smul_smul, hKA, ha]
  rw [h, Matrix.blockDiagonal_one]

omit [DecidableEq d] in
theorem global_inverse_action (a : k → ℝ) (K : k → Matrix d d ℝ)
    (v : d × k → ℝ) (j : d) (b : k) :
    (globalInverse a K *ᵥ v) (j, b) =
      (a b)⁻¹ * (K b *ᵥ (fun i => v (i, b))) j := by
  simp [globalInverse, Matrix.mulVec, dotProduct, Fintype.sum_prod_type,
    Matrix.blockDiagonal_apply, Finset.mul_sum, mul_assoc]

omit [DecidableEq d] in
/-- With row weights `a`, the inverse block factors cancel on a row supported in `b`.
`e` may be the last-coordinate selector or any fixed local row. -/
theorem weight_cancellation (a : k → ℝ) (ha : ∀ j, a j ≠ 0)
    (K : k → Matrix d d ℝ) (e g : d → ℝ) (b : k) :
    (fun jk : d × k => a jk.2 * e jk.1) ⬝ᵥ
      (globalInverse a K *ᵥ (fun jk => if jk.2 = b then g jk.1 else 0)) =
      e ⬝ᵥ (K b *ᵥ g) := by
  simp only [dotProduct, Fintype.sum_prod_type, global_inverse_action]
  simp only [Matrix.mulVec, dotProduct, mul_ite, Finset.sum_ite_irrel]
  simp only [mul_zero, Finset.sum_const_zero, Finset.sum_ite_eq',
    Finset.mem_univ, if_true]
  apply Finset.sum_congr rfl
  intro x hx
  calc
    _ = (a b * (a b)⁻¹) * (e x * ∑ j, K b x j * g j) := by ring
    _ = _ := by rw [mul_inv_cancel₀ (ha b), one_mul]

end GlobalBlocks

section DependentBlocks
variable {k : Type*} {D : k → Type*} [Fintype k] [DecidableEq k]
  [∀ b, Fintype (D b)]

def dependentBread (a : k → ℝ) (A : ∀ b, Matrix (D b) (D b) ℝ) :
    Matrix (Σ b, D b) (Σ b, D b) ℝ :=
  Matrix.blockDiagonal' (fun b => a b • A b)

def dependentInverse (a : k → ℝ) (K : ∀ b, Matrix (D b) (D b) ℝ) :
    Matrix (Σ b, D b) (Σ b, D b) ℝ :=
  Matrix.blockDiagonal' (fun b => (a b)⁻¹ • K b)

theorem dependent_inverse_right [∀ b, DecidableEq (D b)]
    (a : k → ℝ) (ha : ∀ b, a b ≠ 0)
    (A K : ∀ b, Matrix (D b) (D b) ℝ) (hAK : ∀ b, A b * K b = 1) :
    dependentBread a A * dependentInverse a K = 1 := by
  unfold dependentBread dependentInverse
  rw [← Matrix.blockDiagonal'_mul]
  have h : (fun b => (a b • A b) * ((a b)⁻¹ • K b)) =
      (1 : ∀ b, Matrix (D b) (D b) ℝ) := by
    funext b
    simp [smul_smul, hAK, ha]
  rw [h, Matrix.blockDiagonal'_one]

theorem dependent_inverse_left [∀ b, DecidableEq (D b)]
    (a : k → ℝ) (ha : ∀ b, a b ≠ 0)
    (A K : ∀ b, Matrix (D b) (D b) ℝ) (hKA : ∀ b, K b * A b = 1) :
    dependentInverse a K * dependentBread a A = 1 := by
  unfold dependentBread dependentInverse
  rw [← Matrix.blockDiagonal'_mul]
  have h : (fun b => ((a b)⁻¹ • K b) * (a b • A b)) =
      (1 : ∀ b, Matrix (D b) (D b) ℝ) := by
    funext b
    simp [smul_smul, hKA, ha]
  rw [h, Matrix.blockDiagonal'_one]

theorem dependent_inverse_action (a : k → ℝ)
    (K : ∀ b, Matrix (D b) (D b) ℝ) (v : (Σ b, D b) → ℝ) (b : k) (i : D b) :
    (dependentInverse a K *ᵥ v) ⟨b, i⟩ =
      (a b)⁻¹ * (K b *ᵥ (fun j => v ⟨b, j⟩)) i := by
  simp only [dependentInverse, Matrix.mulVec, dotProduct, Fintype.sum_sigma,
    Matrix.blockDiagonal'_apply']
  rw [Fintype.sum_eq_single b]
  · simp [Finset.mul_sum, mul_assoc]
  · intro b' hb'
    apply Finset.sum_eq_zero
    intro j hj
    rw [dif_neg hb'.symm, zero_mul]

theorem dependent_weight_cancellation (a : k → ℝ) (ha : ∀ b, a b ≠ 0)
    (K : ∀ b, Matrix (D b) (D b) ℝ) (e : ∀ b, D b → ℝ)
    (v : (Σ b, D b) → ℝ) :
    (fun x : Σ b, D b => a x.1 * e x.1 x.2) ⬝ᵥ (dependentInverse a K *ᵥ v) =
      ∑ b, e b ⬝ᵥ (K b *ᵥ (fun j => v ⟨b, j⟩)) := by
  simp only [dotProduct, Fintype.sum_sigma, dependent_inverse_action]
  apply Finset.sum_congr rfl
  intro b hb
  apply Finset.sum_congr rfl
  intro i hi
  calc
    _ = (a b * (a b)⁻¹) * (e b i * ∑ j, K b i j * v ⟨b, j⟩) := by
      simp only [Matrix.mulVec, dotProduct]
      ring
    _ = _ := by
      rw [mul_inv_cancel₀ (ha b), one_mul]
      rfl

/-- Arbitrarily different block dimensions, with the actual observation's
single-block support assumed explicitly. No cross-block support is invented. -/
theorem dependent_supported_cancellation (a : k → ℝ) (ha : ∀ b, a b ≠ 0)
    (K : ∀ b, Matrix (D b) (D b) ℝ) (e : ∀ b, D b → ℝ)
    (v : (Σ b, D b) → ℝ) (b : k)
    (hsupport : ∀ b' : k, b' ≠ b → ∀ j : D b', v ⟨b', j⟩ = 0) :
    (fun x : Σ b, D b => a x.1 * e x.1 x.2) ⬝ᵥ (dependentInverse a K *ᵥ v) =
      e b ⬝ᵥ (K b *ᵥ (fun j => v ⟨b, j⟩)) := by
  rw [dependent_weight_cancellation a ha]
  apply Fintype.sum_eq_single b
  intro b' hb'
  simp [hsupport b' hb', Matrix.mulVec, dotProduct]

end DependentBlocks

/-- Heterogeneous nuisance dimensions, actual local joint scores, and the weighted
theta extraction row: the result is exactly the local corrected contribution. -/
theorem dependent_last_coordinate {k : Type*} {P : k → Type*}
    [Fintype k] [DecidableEq k] [∀ b, Fintype (P b)] [∀ b, DecidableEq (P b)]
    (a : k → ℝ) (ha : ∀ b, a b ≠ 0)
    (K : ∀ b, Matrix (P b) (P b) ℝ) (c : ∀ b, Matrix Unit (P b) ℝ)
    (v : (Σ b, P b ⊕ Unit) → ℝ) (b : k)
    (hsupport : ∀ b', b' ≠ b → ∀ j : P b' ⊕ Unit, v ⟨b', j⟩ = 0)
    (g : P b → ℝ) (q theta : ℝ)
    (hlocal : ∀ j, v ⟨b, j⟩ = Sum.elim g (fun _ => q - theta) j) :
    (fun x : Σ b, P b ⊕ Unit =>
      a x.1 * (Pi.single (Sum.inr ()) 1 : P x.1 ⊕ Unit → ℝ) x.2) ⬝ᵥ
      (dependentInverse a (fun b => inverseCandidate (K b) (c b)) *ᵥ v) =
      q - theta + c b () ⬝ᵥ (K b *ᵥ g) := by
  have hv : (fun j => v ⟨b, j⟩) = Sum.elim g (fun _ => q - theta) := funext hlocal
  have hc := dependent_supported_cancellation a ha
    (fun b => inverseCandidate (K b) (c b))
    (fun b => (Pi.single (Sum.inr ()) 1 : P b ⊕ Unit → ℝ)) v b hsupport
  simpa only [single_one_dotProduct, hv, last_coordinate] using hc

#print axioms bread_mul_inverseCandidate
#print axioms inverseCandidate_mul_bread
#print axioms last_coordinate
#print axioms transform_outer
#print axioms outer_quadratic
#print axioms sandwich_quadratic
#print axioms transform_finiteMean
#print axioms transform_centered
#print axioms scalar_transform_centered
#print axioms centered_sandwich_quadratic
#print axioms global_inverse_right
#print axioms global_inverse_left
#print axioms global_inverse_action
#print axioms weight_cancellation
#print axioms dependent_inverse_right
#print axioms dependent_inverse_left
#print axioms dependent_inverse_action
#print axioms dependent_weight_cancellation
#print axioms dependent_supported_cancellation
#print axioms dependent_last_coordinate

end JointSandwich
