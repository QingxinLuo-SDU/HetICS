import BiostatFormal.HighDimensional.Multiview.B2V3Slots
import BiostatFormal.HighDimensional.Multiview.SelectorContract
import BiostatFormal.HighDimensional.Multiview.States

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

universe uOmega uCenter uFit uNorm uFrozen uEval

abbrev B2V3RecoveryRecord
    (Center : Type uCenter) (Fit : Type uFit) (Norm : Type uNorm)
    (Frozen : Type uFrozen) :=
  CandidateRecord Center (Fin 3) B2V3ActiveSlot (RiskContext Fit Norm Frozen)

namespace B2V3RecoveryRecord

def provenance
    {Center : Type uCenter} {Fit : Type uFit} {Norm : Type uNorm}
    {Frozen : Type uFrozen}
    (record : B2V3RecoveryRecord Center Fit Norm Frozen) :
    B2V3ActiveSlot × Direction × CandidateStage :=
  (record.slot, record.direction, record.context.stage)

end B2V3RecoveryRecord

structure B2V3RecoveryContext
    (Omega : Type uOmega) [MeasurableSpace Omega]
    (Center : Type uCenter) (Fit : Type uFit) (Norm : Type uNorm)
    (Frozen : Type uFrozen) (EvaluationContext : Type uEval) where
  jointLaw : Measure Omega
  isProbabilityMeasure : IsProbabilityMeasure jointLaw
  truth : Partition Center (Fin 3)
  guardedPool : Omega → GuardedCandidatePool
    (B2V3RecoveryRecord Center Fit Norm Frozen)
  riskSpec : SelectorRiskSpec
    (B2V3RecoveryRecord Center Fit Norm Frozen) EvaluationContext

namespace B2V3RecoveryContext

variable {Omega : Type uOmega} [MeasurableSpace Omega]
  {Center : Type uCenter} {Fit : Type uFit} {Norm : Type uNorm}
  {Frozen : Type uFrozen} {EvaluationContext : Type uEval}

def recordExact (ctx : B2V3RecoveryContext Omega Center Fit Norm Frozen EvaluationContext)
    (record : B2V3RecoveryRecord Center Fit Norm Frozen) : Prop :=
  Exact record.label ctx.truth

def recordInGuardedPool
    [DecidableEq (B2V3RecoveryRecord Center Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega Center Fit Norm Frozen EvaluationContext)
    (omega : Omega) (record : B2V3RecoveryRecord Center Fit Norm Frozen) : Prop :=
  record ∈ ctx.guardedPool omega

abbrev SelectorOutput
    [DecidableEq (B2V3RecoveryRecord Center Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega Center Fit Norm Frozen EvaluationContext) :=
  (omega : Omega) → AlgorithmOutput (ctx.guardedPool omega) SelectorAbstainReason

def exactSurvival
    [DecidableEq (B2V3RecoveryRecord Center Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega Center Fit Norm Frozen EvaluationContext) : Set Omega :=
  exactSurvivalEvent ctx.guardedPool ctx.truth

def uniformDeviation
    [DecidableEq (B2V3RecoveryRecord Center Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega Center Fit Norm Frozen EvaluationContext)
    (u : Real) : Set Omega :=
  uniformDeviationEvent ctx.guardedPool ctx.riskSpec u

def strictPopulationGap
    [DecidableEq (B2V3RecoveryRecord Center Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega Center Fit Norm Frozen EvaluationContext)
    (u etaTie etaGuard etaNum : Real) : Set Omega :=
  selectorStrictPopulationGapEvent ctx.guardedPool ctx.riskSpec ctx.truth
    u etaTie etaGuard etaNum

def selectorSuccess
    [DecidableEq (B2V3RecoveryRecord Center Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega Center Fit Norm Frozen EvaluationContext)
    (output : ctx.SelectorOutput) (etaTie etaGuard etaNum : Real) : Set Omega :=
  selectorSuccessEvent ctx.guardedPool ctx.riskSpec output ctx.truth etaTie etaGuard etaNum

def exactSelected
    [DecidableEq (B2V3RecoveryRecord Center Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega Center Fit Norm Frozen EvaluationContext)
    (output : ctx.SelectorOutput) : Set Omega :=
  exactSelectedEvent output ctx.truth

theorem semanticWithCoverage
    [Fintype Center] [DecidableEq Center]
    [DecidableEq (B2V3RecoveryRecord Center Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega Center Fit Norm Frozen EvaluationContext)
    (output : ctx.SelectorOutput)
    (u etaTie etaGuard etaNum : Real) (coverageEvent : Set Omega) :
    coverageEvent ∩ ctx.exactSurvival ∩ ctx.uniformDeviation u ∩
        ctx.strictPopulationGap u etaTie etaGuard etaNum ∩
        ctx.selectorSuccess output etaTie etaGuard etaNum ⊆
      ctx.exactSelected output := by
  exact recoverySemanticWithCoverage ctx.guardedPool ctx.riskSpec output ctx.truth
    u etaTie etaGuard etaNum coverageEvent

end B2V3RecoveryContext
end BiostatFormal.HighDimensional.Multiview
