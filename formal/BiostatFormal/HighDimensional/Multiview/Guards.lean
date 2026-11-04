import BiostatFormal.HighDimensional.Multiview.Candidates
import Mathlib.Data.Finset.Basic

/-!
# Guard and abstention states

`REJECT` is local to a guarded candidate.  `ABSTAIN` is the final result when no eligible record is
selected.  Neither constructor contains or fabricates a partition.
-/

namespace BiostatFormal.HighDimensional.Multiview

universe uCandidate uRejectReason uAbstainReason

/-- Informal guarded candidate or `REJECT` ↔ Lean: an explicit per-candidate sum type. -/
inductive GuardResult (Candidate : Type uCandidate) (RejectReason : Type uRejectReason) where
  | accepted (candidate : Candidate)
  | reject (reason : RejectReason)

/-- Informal finite guarded pool ↔ Lean: a finite set containing only accepted candidates. -/
abbrev GuardedCandidatePool (Candidate : Type uCandidate) := Finset Candidate

/--
Informal `z ∈ P_MV^guard` ↔ Lean: a candidate bundled with proof that it belongs to the realized
finite guarded pool.  Using this subtype as the candidate type prevents risk assumptions from
silently ranging over rejected or otherwise absent records.
-/
abbrev GuardedPoolMember {Candidate : Type uCandidate} [DecidableEq Candidate]
    (pool : GuardedCandidatePool Candidate) :=
  {candidate // candidate ∈ pool}

/--
Informal selected guarded record or global `ABSTAIN` ↔ Lean: a final output indexed by the
realized guarded pool.  The selected constructor carries its pool-membership proof, while the
abstention reason is retained for the source-required audit trail.
-/
inductive AlgorithmOutput
    {Candidate : Type uCandidate} [DecidableEq Candidate]
    (pool : GuardedCandidatePool Candidate) (AbstainReason : Type uAbstainReason) where
  | selected (candidate : GuardedPoolMember pool)
  | abstain (reason : AbstainReason)

/-- Every selected algorithm output is definitionally backed by membership in its guarded pool. -/
@[simp]
theorem AlgorithmOutput.selected_mem
    {Candidate : Type uCandidate} [DecidableEq Candidate]
    {pool : GuardedCandidatePool Candidate}
    (candidate : GuardedPoolMember pool) :
    candidate.1 ∈ pool :=
  candidate.2

/--
Compatibility name for the final selection state.  Unlike the earlier unindexed version, callers
must supply the realized pool, so a selected nonmember is unrepresentable.
-/
abbrev SelectionDecision
    {Candidate : Type uCandidate} [DecidableEq Candidate]
    (pool : GuardedCandidatePool Candidate) (AbstainReason : Type uAbstainReason) :=
  AlgorithmOutput pool AbstainReason

end BiostatFormal.HighDimensional.Multiview
