import Mathlib.Data.Fintype.Basic
import Mathlib.Tactic.DeriveFintype

/-!
# Candidate-generation views

The seven source views index candidate generation.  They are deliberately distinct from the four
components of the composite validation risk, which are defined in `Risks.lean`.
-/

namespace BiostatFormal.HighDimensional.Multiview

/-- Informal `{X,Y,F,XY,XF,YF,XYF}` ↔ Lean: seven candidate-generation views. -/
inductive View where
  | treatment
  | outcome
  | covariate
  | treatmentOutcome
  | treatmentCovariate
  | outcomeCovariate
  | treatmentOutcomeCovariate
  deriving DecidableEq, Fintype, Repr

/--
Informal eligible view ↔ Lean: a view bundled with evidence for a caller-supplied, possibly
context-dependent eligibility predicate.  No determinism or data independence is asserted here.
-/
structure EligibleView (Eligible : View → Prop) where
  view : View
  eligible : Eligible view

end BiostatFormal.HighDimensional.Multiview
