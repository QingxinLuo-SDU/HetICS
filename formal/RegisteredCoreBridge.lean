import HierarchicalEffects
import ScreeningAdjustment

/-! Application bridge from the registered binary-DGP core to the final
post-ICP adjustment set and its classwise AIPW causal target. -/

noncomputable section
open scoped BigOperators
open MeasureTheory
open BinaryFormal.HierarchicalEffects
open BinaryFormal.ScreeningAdjustment
open BiostatFormal.HighDimensional.KnockoffAdjustment

namespace BinaryFormal.RegisteredCoreBridge

variable {Omega I K V : Type*} [DecidableEq I] [DecidableEq K]
  [Fintype K] [DecidableEq V]

/-- The paper's application mechanism uses exactly Z, shared causes and the
own-class specific variables as its active adjustment core. -/
def registeredMechanism (Z shared specific : Finset V) (effect : Real) :
    ClusterMechanism V where
  active := registeredCore Z shared specific
  effect := effect

theorem registeredMechanism_active (Z shared specific : Finset V) (effect : Real) :
    (registeredMechanism Z shared specific effect).active =
      registeredCore Z shared specific := rfl

theorem registeredMechanism_effect (Z shared specific : Finset V) (effect : Real) :
    (registeredMechanism Z shared specific effect).effect = effect := rfl

/-- Coverage and safety of the actual post-ICP set now consume the registered
core, instead of a disconnected generic set. -/
theorem registered_final_valid (Z shared specific finalSet : Finset V)
    (safe : Finset V -> Prop)
    (coverage : registeredCore Z shared specific <= finalSet)
    (safety : safe finalSet) :
    ValidAdjustment (registeredMechanism Z shared specific 0).active safe finalSet := by
  exact final_valid_of_coverage_safe _ _ safe coverage safety

section Identification

variable [MeasurableSpace Omega]

/-- Single-class endpoint of the application bridge.  The mechanism's active
set qualifies the final set; the supplied causal bridge then identifies the
mechanism's risk-difference effect. -/
theorem registered_final_aipw_identifies_effect
    (mu : Measure Omega) (Y0 Y1 : Omega -> Real)
    (hY0 : Integrable Y0 mu) (hY1 : Integrable Y1 mu)
    (Z shared specific finalSet : Finset V) (effect : Real)
    (safe : Finset V -> Prop)
    (bridge : CommonAdjustmentBridge mu Y0 Y1
      (registeredMechanism Z shared specific effect).active safe)
    (coverage : (registeredMechanism Z shared specific effect).active <= finalSet)
    (safety : safe finalSet)
    (effect_target : effect = commonATE mu Y0 Y1)
    (e m0 m1 : Omega -> Real)
    (domain : ∀ᵐ x ∂mu, 0 < e x ∧ e x < 1)
    (correct : (∀ᵐ x ∂mu, e x = (bridge.lawFor finalSet x).e0) ∨
      (∀ᵐ x ∂mu, m0 x = (bridge.lawFor finalSet x).mu0 ∧
        m1 x = (bridge.lawFor finalSet x).mu1)) :
    populationAIPW mu (bridge.lawFor finalSet) e m0 m1 = effect := by
  rw [effect_target]
  exact aipw_identifies_common_ate mu Y0 Y1 hY0 hY1 _
    (bridge.conditions_of_valid finalSet
      (final_valid_of_coverage_safe _ _ safe coverage safety))
    e m0 m1 domain correct

/-- Classwise registered mechanisms, final-set qualification and AIPW
identification reach the actual person-count weighted effect target. -/
theorem registered_person_weighted_aipw_identifies_effects
    (p : BinaryCore.SamplePartition I K)
    (mu : K -> Measure Omega) (Y0 Y1 : K -> Omega -> Real)
    (hY0 : forall k, Integrable (Y0 k) (mu k))
    (hY1 : forall k, Integrable (Y1 k) (mu k))
    (Z shared : Finset V) (specific finalSet : K -> Finset V)
    (effect : K -> Real) (safe : K -> Finset V -> Prop)
    (bridge : forall k, CommonAdjustmentBridge (mu k) (Y0 k) (Y1 k)
      (registeredMechanism Z shared (specific k) (effect k)).active (safe k))
    (coverage : forall k,
      (registeredMechanism Z shared (specific k) (effect k)).active <= finalSet k)
    (safety : forall k, safe k (finalSet k))
    (effect_target : forall k, effect k = commonATE (mu k) (Y0 k) (Y1 k))
    (e m0 m1 : K -> Omega -> Real)
    (domain : ∀ k, ∀ᵐ x ∂mu k, 0 < e k x ∧ e k x < 1)
    (correct : forall k,
      (∀ᵐ x ∂mu k, e k x = ((bridge k).lawFor (finalSet k) x).e0) ∨
      (∀ᵐ x ∂mu k, m0 k x = ((bridge k).lawFor (finalSet k) x).mu0 ∧
        m1 k x = ((bridge k).lawFor (finalSet k) x).mu1)) :
    ((∑ k, p.weight k * populationAIPW (mu k)
      ((bridge k).lawFor (finalSet k)) (e k) (m0 k) (m1 k)) =
      ∑ k, p.weight k * effect k) := by
  apply Finset.sum_congr rfl
  intro k _
  rw [registered_final_aipw_identifies_effect (mu k) (Y0 k) (Y1 k)
    (hY0 k) (hY1 k) Z shared (specific k) (finalSet k) (effect k)
    (safe k) (bridge k) (coverage k) (safety k) (effect_target k)
    (e k) (m0 k) (m1 k) (domain k) (correct k)]

end Identification

#print axioms registeredMechanism
#print axioms registeredMechanism_active
#print axioms registeredMechanism_effect
#print axioms registered_final_valid
#print axioms registered_final_aipw_identifies_effect
#print axioms registered_person_weighted_aipw_identifies_effects

end BinaryFormal.RegisteredCoreBridge
