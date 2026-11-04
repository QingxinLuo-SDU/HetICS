import BinaryCore
import JointSandwich
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FinCases

/-! Actual real logistic derivatives, fixed design, and empirical joint bread.
No claim of automatic invertibility, regular roots, or valid standard errors. -/
noncomputable section
open scoped BigOperators
open Matrix
namespace AIPWJacobian

def expit (z : ℝ) : ℝ := Real.exp z / (1 + Real.exp z)

theorem expit_range (z : ℝ) : 0 < expit z ∧ expit z < 1 := by
  have he := Real.exp_pos z
  constructor
  · exact div_pos he (by positivity)
  · exact (div_lt_one (by positivity)).2 (by linarith)

theorem expit_ne_zero (z : ℝ) : expit z ≠ 0 := ne_of_gt (expit_range z).1
theorem one_sub_expit_ne_zero (z : ℝ) : 1 - expit z ≠ 0 :=
  ne_of_gt (sub_pos.mpr (expit_range z).2)

theorem expit_eq_inverse_form (z : ℝ) : expit z = 1 / (1 + Real.exp (-z)) := by
  rw [Real.exp_neg]
  unfold expit
  have he : Real.exp z ≠ 0 := ne_of_gt (Real.exp_pos z)
  have hp : 1 + Real.exp z ≠ 0 := by positivity
  field_simp [he, hp]
  ring

theorem hasDerivAt_expit (z : ℝ) :
    HasDerivAt expit (expit z * (1 - expit z)) z := by
  have hn : 1 + Real.exp z ≠ 0 := by positivity
  convert (Real.hasDerivAt_exp z).div ((hasDerivAt_const z (1 : ℝ)).add
    (Real.hasDerivAt_exp z)) hn using 1 <;> try rfl
  dsimp [expit]
  field_simp [hn]
  ring

theorem hasDerivAt_expit_affine (z d t : ℝ) :
    HasDerivAt (fun s => expit (z + s * d))
      (expit (z + t*d) * (1-expit (z+t*d)) * d) t := by
  convert (hasDerivAt_expit (z+t*d)).comp t
    ((hasDerivAt_const t z).add ((hasDerivAt_id t).mul_const d)) using 1 <;> try rfl
  ring

def ca (x y e m0 m1 : ℝ) : ℝ :=
  -(x*(y-m1)*(1-e)/e + (1-x)*(y-m0)*e/(1-e))
def cb0 (x e m0 : ℝ) : ℝ := m0*(1-m0)*((1-x)/(1-e)-1)
def cb1 (x e m1 : ℝ) : ℝ := m1*(1-m1)*(1-x/e)

/-- All three nuisance directions can vary simultaneously. -/
theorem hasDerivAt_phi_path (x y a b0 b1 da db0 db1 : ℝ) :
    HasDerivAt (fun t => BinaryCore.phi x y (expit (a+t*da))
      (expit (b0+t*db0)) (expit (b1+t*db1)))
      (ca x y (expit a) (expit b0) (expit b1)*da +
       cb0 x (expit a) (expit b0)*db0 + cb1 x (expit a) (expit b1)*db1) 0 := by
  have he := hasDerivAt_expit_affine a da 0
  have h0 := hasDerivAt_expit_affine b0 db0 0
  have h1 := hasDerivAt_expit_affine b1 db1 0
  simp only [zero_mul, add_zero] at he h0 h1
  have h := ((h1.sub h0).add
    (((hasDerivAt_const 0 x).div he (by simpa using expit_ne_zero a)).mul
      ((hasDerivAt_const 0 y).sub h1))).sub
    (((hasDerivAt_const 0 (1-x)).div ((hasDerivAt_const 0 (1:ℝ)).sub he)
      (by simpa using one_sub_expit_ne_zero a)).mul ((hasDerivAt_const 0 y).sub h0))
  convert h using 1 <;> try rfl
  dsimp [ca, cb0, cb1]
  simp only [zero_mul, add_zero]
  field_simp [expit_ne_zero a, one_sub_expit_ne_zero a]
  ring

theorem hasDerivAt_score_path (w r v z d : ℝ) :
    HasDerivAt (fun t => w*r*(v-expit (z+t*d)))
      (-w*r*expit z*(1-expit z)*d) 0 := by
  convert (((hasDerivAt_const 0 v).sub
    (hasDerivAt_expit_affine z d 0)).const_mul (w*r)) using 1 <;> try rfl
  simp only [zero_mul, add_zero, zero_sub]
  ring

section FixedDesign
variable {p : Type*} [Fintype p] [DecidableEq p]

def linear (r : p → ℝ) (η : Fin 3 → p → ℝ) (b : Fin 3) : ℝ := ∑ j, r j * η b j
def move (η : Fin 3 → p → ℝ) (c : Fin 3 × p) (t : ℝ) : Fin 3 → p → ℝ :=
  fun b j => η b j + if b = c.1 ∧ j = c.2 then t else 0

theorem linear_move (r : p → ℝ) (η : Fin 3 → p → ℝ)
    (c : Fin 3 × p) (b : Fin 3) (t : ℝ) :
    linear r (move η c t) b = linear r η b + t * (if b = c.1 then r c.2 else 0) := by
  unfold linear move
  rw [Finset.sum_congr rfl (fun j _ => mul_add (r j) (η b j) _), Finset.sum_add_distrib]
  by_cases h : b = c.1
  · simp [h, mul_ite, Finset.sum_ite_eq', mul_comm]
  · simp [h]

def weight (x : ℝ) (b : Fin 3) : ℝ := if b = 0 then 1 else if b = 1 then 1-x else x
def response (x y : ℝ) (b : Fin 3) : ℝ := if b = 0 then x else y
def score (r : p → ℝ) (x y : ℝ) (η : Fin 3 → p → ℝ) (a : Fin 3 × p) : ℝ :=
  weight x a.1 * r a.2 * (response x y a.1 - expit (linear r η a.1))
def target (r : p → ℝ) (x y : ℝ) (η : Fin 3 → p → ℝ) : ℝ :=
  BinaryCore.phi x y (expit (linear r η 0)) (expit (linear r η 1)) (expit (linear r η 2))
def grad (r : p → ℝ) (x y : ℝ) (η : Fin 3 → p → ℝ) (c : Fin 3 × p) : ℝ :=
  r c.2 * (if c.1 = 0 then ca x y (expit (linear r η 0))
    (expit (linear r η 1)) (expit (linear r η 2))
    else if c.1 = 1 then cb0 x (expit (linear r η 0)) (expit (linear r η 1))
    else cb1 x (expit (linear r η 0)) (expit (linear r η 2)))
def info (r : p → ℝ) (x : ℝ) (η : Fin 3 → p → ℝ) : Matrix (Fin 3 × p) (Fin 3 × p) ℝ :=
  fun a c => if a.1 = c.1 then weight x a.1 * expit (linear r η a.1) *
    (1-expit (linear r η a.1)) * r a.2 * r c.2 else 0

/-- Every nuisance entry: includes zero derivatives between different models. -/
theorem score_coordinate_derivative (r : p → ℝ) (x y : ℝ) (η : Fin 3 → p → ℝ)
    (a c : Fin 3 × p) :
    HasDerivAt (fun t => score r x y (move η c t) a) (-info r x η a c) 0 := by
  simp only [score, linear_move]
  convert hasDerivAt_score_path (weight x a.1) (r a.2) (response x y a.1)
    (linear r η a.1) (if a.1 = c.1 then r c.2 else 0) using 1
  dsimp [info]
  split_ifs <;> ring

/-- Actual AIPW gradient in each alpha/beta0/beta1 coordinate. -/
theorem target_coordinate_derivative (r : p → ℝ) (x y : ℝ) (η : Fin 3 → p → ℝ)
    (c : Fin 3 × p) :
    HasDerivAt (fun t => target r x y (move η c t)) (grad r x y η c) 0 := by
  simp only [target, linear_move]
  have h := hasDerivAt_phi_path x y (linear r η 0) (linear r η 1) (linear r η 2)
    (if (0:Fin 3) = c.1 then r c.2 else 0)
    (if (1:Fin 3) = c.1 then r c.2 else 0)
    (if (2:Fin 3) = c.1 then r c.2 else 0)
  convert h using 1
  rcases c with ⟨b,j⟩
  fin_cases b <;> simp [grad] <;> ring

def joint (r : p → ℝ) (x y : ℝ) (η : Fin 3 → p → ℝ) (θ : ℝ) :
    (Fin 3 × p) ⊕ Unit → ℝ := Sum.elim (score r x y η) (fun _ => target r x y η - θ)

/-- Parameter column path includes theta; fixed design and observed data. -/
def jointPath (r : p → ℝ) (x y : ℝ) (η : Fin 3 → p → ℝ) (θ : ℝ)
    (c : (Fin 3 × p) ⊕ Unit) (t : ℝ) : (Fin 3 × p) ⊕ Unit → ℝ :=
  match c with
  | .inl c => joint r x y (move η c t) θ
  | .inr _ => joint r x y η (θ+t)

theorem joint_coordinate_derivative (r : p → ℝ) (x y : ℝ) (η : Fin 3 → p → ℝ)
    (θ : ℝ) (a c : (Fin 3 × p) ⊕ Unit) :
    HasDerivAt (fun t => jointPath r x y η θ c t a)
      (-(JointSandwich.bread (info r x η) (fun _ => grad r x y η)) a c) 0 := by
  cases a with
  | inl a =>
    cases c with
    | inl c => exact score_coordinate_derivative r x y η a c
    | inr c => simpa [jointPath, joint, JointSandwich.bread] using hasDerivAt_const 0 (score r x y η a)
  | inr a =>
    cases c with
    | inl c => simpa [jointPath, joint, JointSandwich.bread] using
        (target_coordinate_derivative r x y η c).sub_const θ
    | inr c =>
      convert (hasDerivAt_const 0 (target r x y η)).sub
        ((hasDerivAt_const 0 θ).add (hasDerivAt_id 0)) using 1 <;> try rfl
      simp [JointSandwich.bread]

variable {ι : Type*} [Fintype ι]
def meanInfo (r : ι → p → ℝ) (x : ι → ℝ) (η : Fin 3 → p → ℝ) :
    Matrix (Fin 3 × p) (Fin 3 × p) ℝ :=
  fun a c => (Fintype.card ι : ℝ)⁻¹ * ∑ i, info (r i) (x i) η a c
def meanGrad (r : ι → p → ℝ) (x y : ι → ℝ) (η : Fin 3 → p → ℝ) :
    Matrix Unit (Fin 3 × p) ℝ :=
  fun _ c => (Fintype.card ι : ℝ)⁻¹ * ∑ i, grad (r i) (x i) (y i) η c
def actualBread (r : ι → p → ℝ) (x y : ι → ℝ) (η : Fin 3 → p → ℝ) (θ : ℝ) :
    Matrix ((Fin 3 × p) ⊕ Unit) ((Fin 3 × p) ⊕ Unit) ℝ :=
  fun a c => -((Fintype.card ι : ℝ)⁻¹ * ∑ i, deriv (fun t => jointPath (r i) (x i) (y i) η θ c t a) 0)

/-- Actual negative average of derivatives, not a stipulated Jacobian. -/
theorem actualBread_eq [Nonempty ι] (r : ι → p → ℝ) (x y : ι → ℝ)
    (η : Fin 3 → p → ℝ) (θ : ℝ) :
    actualBread r x y η θ = JointSandwich.bread (meanInfo r x η) (meanGrad r x y η) := by
  have hn : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  ext a c
  simp only [actualBread, (joint_coordinate_derivative _ _ _ _ _ _ _).deriv]
  cases a <;> cases c <;>
    simp [JointSandwich.bread, meanInfo, meanGrad, Finset.sum_neg_distrib, hn]

/-- Consumer calls the derivative-to-bread theorem and existing inverse algebra. -/
theorem actual_inverse_and_contribution [Nonempty ι]
    (r : ι → p → ℝ) (x y : ι → ℝ) (η : Fin 3 → p → ℝ) (θ : ℝ)
    (K : Matrix (Fin 3 × p) (Fin 3 × p) ℝ)
    (hK : K * meanInfo r x η = 1) (i : ι) :
    JointSandwich.inverseCandidate K (meanGrad r x y η) * actualBread r x y η θ = 1 ∧
    (JointSandwich.inverseCandidate K (meanGrad r x y η) *ᵥ joint (r i) (x i) (y i) η θ) (.inr ()) =
      target (r i) (x i) (y i) η - θ + meanGrad r x y η () ⬝ᵥ (K *ᵥ score (r i) (x i) (y i) η) := by
  constructor
  · rw [actualBread_eq]
    exact JointSandwich.inverseCandidate_mul_bread _ _ _ hK
  · exact JointSandwich.last_coordinate _ _ _ _ _

end FixedDesign

#print axioms linear_move
#print axioms score_coordinate_derivative
#print axioms target_coordinate_derivative
#print axioms joint_coordinate_derivative
#print axioms actualBread_eq
#print axioms actual_inverse_and_contribution
#print axioms expit_range
#print axioms expit_eq_inverse_form
#print axioms expit_ne_zero
#print axioms one_sub_expit_ne_zero
#print axioms hasDerivAt_expit
#print axioms hasDerivAt_expit_affine
#print axioms hasDerivAt_phi_path
#print axioms hasDerivAt_score_path
end AIPWJacobian
