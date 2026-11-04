import BiostatFormal.HighDimensional.Multiview.Candidates
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Real.Basic

/-!
# Population and empirical composite risks

The four risk components are not candidate-generation views.  A `CompositeRiskPair` shares one
weight function between its population and empirical components, mirroring (MV14)--(MV15) while
leaving expectation, measurability, and sample splitting to the later probability layer.
-/

namespace BiostatFormal.HighDimensional.Multiview

universe uCandidate

/--
Marker for complete objects that may be passed to the multiview risk API.  Contextual records are
risk inputs; transient states such as `CandidateState` deliberately receive no instance.
-/
class RiskInput (Candidate : Type uCandidate) : Prop where

/-- Complete contextual candidate records are admissible risk inputs. -/
instance candidateRecordRiskInput
    {Center : Type*} {Label : Type*} {Slot : Type*} {Context : Type*} :
    RiskInput (CandidateRecord Center Label Slot Context) :=
  ⟨⟩

/-- A proof-carrying subtype of an admissible risk input remains admissible. -/
instance subtypeRiskInput
    {Candidate : Type uCandidate} [RiskInput Candidate] (Legal : Candidate → Prop) :
    RiskInput {candidate // Legal candidate} :=
  ⟨⟩

/-- Informal `Y`, `X`, `F`, and stability risk terms ↔ Lean: four risk-component indices. -/
inductive RiskComponent where
  | outcome
  | treatment
  | covariate
  | stability
  deriving DecidableEq, Fintype, Repr

/--
Informal population/empirical component risks with fixed shared weights ↔ Lean: two record-level
component functions and one common weight function.  Source-side weight constraints remain
explicit assumptions for later theorems rather than fields silently asserted by this structure.
-/
structure CompositeRiskPair (Candidate : Type uCandidate) [RiskInput Candidate] where
  weight : RiskComponent → ℝ
  populationComponent : RiskComponent → Candidate → ℝ
  empiricalComponent : RiskComponent → Candidate → ℝ

/-- Informal population composite risk `R_MV` ↔ Lean: the weighted finite component sum. -/
def populationCompositeRisk {Candidate : Type uCandidate} [RiskInput Candidate]
    (risk : CompositeRiskPair Candidate) (candidate : Candidate) : ℝ :=
  ∑ component : RiskComponent, risk.weight component * risk.populationComponent component candidate

/-- Informal empirical composite risk `Rhat_MV` ↔ Lean: the same weighted finite component sum. -/
def empiricalCompositeRisk {Candidate : Type uCandidate} [RiskInput Candidate]
    (risk : CompositeRiskPair Candidate) (candidate : Candidate) : ℝ :=
  ∑ component : RiskComponent, risk.weight component * risk.empiricalComponent component candidate

end BiostatFormal.HighDimensional.Multiview
