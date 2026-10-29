// ---------------------------------------------------------------------------
// Local_DsG4OpAbsReemit
// Optical absorption and re-emission process for the JUNO New Optical Model
// PLUS.  Three instances (kSolvent, kPPO, kbisMSB) are registered in the
// physics list; Geant4 selects the appropriate one by sampling absorption
// length.
//
// On absorption, the process decides whether to re-emit based on the
// re-emission probability (QE) of the material.  The re-emission energy is
// sampled from the emission spectrum CDF.
// ---------------------------------------------------------------------------
// Author: Tao Lin <lintao@ihep.ac.cn>
//         Xuhui Yang, Blyth Simon
// Refs:   G4OpWLS.hh, G4Absorption.hh
// ---------------------------------------------------------------------------

#ifndef Local_DsG4OpAbsReemit_h
#define Local_DsG4OpAbsReemit_h

#include "G4VDiscreteProcess.hh"
#include "G4PhysicsTable.hh"
#include "G4PhysicsOrderedFreeVector.hh"
#include "G4OpticalPhoton.hh"
#include "U4_API_EXPORT.hh"

class U4_API Local_DsG4OpAbsReemit: public G4VDiscreteProcess
{
public:
    enum class ReemitComponent {
        kSolvent,  // Main solvent (LAB) absorption
        kPPO,      // PPO solute absorption
        kbisMSB    // bis-MSB solute absorption
    };

    Local_DsG4OpAbsReemit(ReemitComponent comp = ReemitComponent::kPPO,
                    const G4String& processName = "OpAbsReemit",
                    G4ProcessType type = fOptical);
    ~Local_DsG4OpAbsReemit();

public:
    G4bool IsApplicable(const G4ParticleDefinition& aParticleType);

    G4double GetMeanFreePath(const G4Track& aTrack, G4double, G4ForceCondition*);

    G4VParticleChange* PostStepDoIt(const G4Track& aTrack, const G4Step& aStep);

    G4PhysicsTable* GetIntegralTable() const;

    void DumpPhysicsTable() const;

    void SetOpticksMode(G4int opticksMode) { m_opticksMode = opticksMode; }
    G4int GetOpticksMode() const { return m_opticksMode; }

private:
    void BuildThePhysicsTable();

    // CDF tables for Inverse Transform Sampling (shared with Local_DsG4ScintSimple)
    G4PhysicsTable* thePPOCDFTable;
    G4PhysicsTable* theBisCDFTable;

    ReemitComponent m_comp;

    G4String m_label_abslen;

    G4int m_ls_material_index;

    // Non-Radiative (NR) transfer parameters
    G4double m_prob_lab2ppo;
    G4double m_prob_lab2bis;
    G4double m_tau_lab2ppo;
    G4double m_tau_lab2bis;

    // Intrinsic emission delays
    G4double m_tau_intrinsic_ppo;
    G4double m_tau_intrinsic_bis;

    // Pre-registered micro-process indices for track tagging.
    // Set in the constructor via GetMicroProcessIndex; used in the hot path
    // via SetModelTag — no string operations, no registry lookups.
    G4int m_procIndex_PPO_LAB;   // "AbsRe_LAB_NR_PPO"
    G4int m_procIndex_Bis_LAB;   // "AbsRe_LAB_NR_bisMSB"
    G4int m_procIndex_PPO;       // "AbsRe_PPO"
    G4int m_procIndex_Bis;       // "AbsRe_bisMSB"

    // Cached re-emission probability vectors (LS material only)
    G4MaterialPropertyVector* m_reemiprob_ppo_vec;
    G4MaterialPropertyVector* m_reemiprob_bis_vec;

    G4int m_opticksMode = 0;
};

inline G4bool Local_DsG4OpAbsReemit::IsApplicable(const G4ParticleDefinition& aParticleType) {
    return (&aParticleType == G4OpticalPhoton::OpticalPhoton());
}

inline G4PhysicsTable* Local_DsG4OpAbsReemit::GetIntegralTable() const {
    return thePPOCDFTable;
}

#ifndef NDEBUG
inline void Local_DsG4OpAbsReemit::DumpPhysicsTable() const {
    if (thePPOCDFTable) {
        G4int n = thePPOCDFTable->entries();
        for (G4int i = 0; i < n; ++i) {
            auto* v = (G4PhysicsOrderedFreeVector*)(*thePPOCDFTable)[i];
            if (v) v->DumpValues();
        }
    }
    if (theBisCDFTable) {
        G4int n = theBisCDFTable->entries();
        for (G4int i = 0; i < n; ++i) {
            auto* v = (G4PhysicsOrderedFreeVector*)(*theBisCDFTable)[i];
            if (v) v->DumpValues();
        }
    }
}
#endif // !NDEBUG

#endif