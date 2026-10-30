// ---------------------------------------------------------------------------
// Local_DsG4ScintSimple
// Primary scintillation process for the JUNO New Optical Model PLUS.
// Derived from G4Scintillation; heavily modified for the JUNO LS system.
//
// Features:
//   1. Birks' law ionization quenching
//   2. Singlet/Triplet (Fast/Slow) mixing, particle-dependent for bis-MSB
//   3. Error Diffusion for exact photon yield conservation
//   4. Inverse Transform Sampling (CDF) for PPO & bis-MSB emission spectra
//   5. Shared CDF tables across processes (DsOpticalModelHelper)
// ---------------------------------------------------------------------------
// Authors: Liang Zhan, bv@bnl.gov, Xuhui Yang, Blyth Simon
// ---------------------------------------------------------------------------

#ifndef Local_DsG4ScintSimple_h
#define Local_DsG4ScintSimple_h 1

#include "G4VRestDiscreteProcess.hh"
#include "G4PhysicsTable.hh"
#include "G4PhysicsOrderedFreeVector.hh"
#include "U4_API_EXPORT.hh"
#include <map>
#include <array>

class U4_API Local_DsG4ScintSimple : public G4VRestDiscreteProcess
{
public:
    // Component index for the 4-component scintillation emission.
    // Keep in order: PPO Fast/Slow, bis-MSB Fast/Slow — used as array indices.
    enum ScintComponent : G4int {
        kPPO_Fast = 0,
        kPPO_Slow,
        kBis_Fast,
        kBis_Slow,
        kNumComponents
    };

    // Particle type for bis-MSB particle-dependent parameters.
    // kElectron is the default for e-, e+, and all unknown particles.
    enum class BisParticle : G4int {
        kElectron = 0,
        kGamma,
        kAlpha,
        kProton,
        kNeutron,
        kNumTypes
    };

    // Birks constant index — maps PDG groups to compact array indices.
    // kEe = electron/positron/gamma, kP = proton/neutron, kA = alpha.
    enum class BirksIdx : G4int {
        kEe = 0,
        kP,
        kA,
        kNumBirks
    };

private:
    // Pre-registered integer indices for the four scintillation components.
    // Set in the constructor via GetMicroProcessIndex; used in the hot path
    // via SetModelTag — no string operations, no registry lookups.
    std::array<G4int, kNumComponents> m_procIndex;

public:
    Local_DsG4ScintSimple(const G4String& processName = "Scintillation",
                      G4ProcessType type = fElectromagnetic);
    ~Local_DsG4ScintSimple();

    G4bool IsApplicable(const G4ParticleDefinition& aParticleType);
    G4double GetMeanFreePath(const G4Track& aTrack, G4double, G4ForceCondition*);
    G4double GetMeanLifeTime(const G4Track& aTrack, G4ForceCondition*);

    G4VParticleChange* PostStepDoIt(const G4Track& aTrack, const G4Step& aStep);
    G4VParticleChange* AtRestDoIt(const G4Track& aTrack, const G4Step& aStep);

    void SetTrackSecondariesFirst(const G4bool state);
    G4bool GetTrackSecondariesFirst() const;

    void SetScintillationYieldFactor(const G4double yieldfactor);
    G4double GetScintillationYieldFactor() const;

    void SetUseFastMu300nsTrick(const G4bool fastMu300nsTrick);
    G4bool GetUseFastMu300nsTrick() const;

    G4PhysicsTable* GetPPOCDFTable() const;
    G4PhysicsTable* GetBisCDFTable() const;

    void DumpPhysicsTable() const;

    G4double GetPhotonWeight() const { return fPhotonWeight; }
    void SetPhotonWeight(G4double weight) { fPhotonWeight = weight; }

    void SetDoQuenching(const G4bool enable) { fEnableQuenching = enable; }
    G4bool GetDoQuenching() const { return fEnableQuenching; }

    G4bool GetApplyPreQE() const { return fApplyPreQE; }
    void SetApplyPreQE(G4bool a) { fApplyPreQE = a; }

    G4double GetPreQE() const { return fPreQE; }
    void SetPreQE(G4double a) { fPreQE = a; }

    void SetBirksConstant1(const std::map<int, double> &m1) {
        birks1[static_cast<G4int>(BirksIdx::kEe)] = m1.at(11);
        birks1[static_cast<G4int>(BirksIdx::kP)]  = m1.at(2212);
        birks1[static_cast<G4int>(BirksIdx::kA)]  = m1.at(1000020040);
    }
    const std::array<G4double, static_cast<G4int>(BirksIdx::kNumBirks)>& GetBirksConstant1() const { return birks1; }
    void SetBirksConstant2(const std::map<int, double> &m2) {
        birks2[static_cast<G4int>(BirksIdx::kEe)] = m2.at(11);
        birks2[static_cast<G4int>(BirksIdx::kP)]  = m2.at(2212);
        birks2[static_cast<G4int>(BirksIdx::kA)]  = m2.at(1000020040);
    }
    const std::array<G4double, static_cast<G4int>(BirksIdx::kNumBirks)>& GetBirksConstant2() const { return birks2; }

    void SetFlagDecayTimeFast(bool flag) { flagDecayTimeFast = flag; }
    bool GetFlagDecayTimeFast() { return flagDecayTimeFast; }
    void SetFlagDecayTimeSlow(bool flag) { flagDecayTimeSlow = flag; }
    bool GetFlagDecayTimeSlow() { return flagDecayTimeSlow; }

    void SetNoOp(bool tf = true) { m_noop = tf; }

    void SetOpticksMode(G4int opticksMode) { m_opticksMode = opticksMode; }
    G4int GetOpticksMode() const { return m_opticksMode; }

private:
    void BuildThePhysicsTable();

    // CDF tables for Inverse Transform Sampling (shared with Local_DsG4OpAbsReemit)
    G4PhysicsTable* thePPOCDFTable;
    G4PhysicsTable* theBisCDFTable;

    G4bool fEnableQuenching;

    // Birks constants arrays indexed by BirksIdx enum
    std::array<G4double, static_cast<G4int>(BirksIdx::kNumBirks)> birks1;
    std::array<G4double, static_cast<G4int>(BirksIdx::kNumBirks)> birks2;

    bool flagDecayTimeFast;
    bool flagDecayTimeSlow;

    G4bool fTrackSecondariesFirst;
    G4double YieldFactor;
    G4bool FastMu300nsTrick;
    G4double fPhotonWeight;
    G4bool   fApplyPreQE;
    G4double fPreQE;
    bool m_noop;

    G4int m_opticksMode = 0;

    // Error Diffusion (Residual Accumulation) — fractional remainders
    G4double m_rem_ppo;
    G4double m_rem_bis;
    G4double m_rem_ppo_fast;
    G4double m_rem_bis_fast;

    // Cached LS optical parameters
    G4int    m_ls_material_index;
    G4double m_scintillation_yield;
    G4double m_resolution_scale;

    // NR transfer probabilities and delays
    G4double m_prob_lab2ppo;
    G4double m_prob_lab2bis;
    G4double m_tau_lab2ppo;
    G4double m_tau_lab2bis;

    // PPO constants
    G4double m_ratio_ppo;
    G4double m_tau_ppo_slow;
    G4double m_tau_intrinsic_ppo;

    // bis-MSB particle-dependent constants (indexed by BisParticle enum)
    G4double m_tau_intrinsic_bis;
    std::array<G4double, static_cast<G4int>(BisParticle::kNumTypes)> m_ratio_bis;
    std::array<G4double, static_cast<G4int>(BisParticle::kNumTypes)> m_tauslow_bis;
};

// Inline methods
inline void Local_DsG4ScintSimple::SetTrackSecondariesFirst(const G4bool state) { fTrackSecondariesFirst = state; }
inline G4bool Local_DsG4ScintSimple::GetTrackSecondariesFirst() const { return fTrackSecondariesFirst; }
inline void Local_DsG4ScintSimple::SetScintillationYieldFactor(const G4double yieldfactor) { YieldFactor = yieldfactor; }
inline G4double Local_DsG4ScintSimple::GetScintillationYieldFactor() const { return YieldFactor; }
inline void Local_DsG4ScintSimple::SetUseFastMu300nsTrick(const G4bool fastMu300nsTrick) { FastMu300nsTrick = fastMu300nsTrick; }
inline G4bool Local_DsG4ScintSimple::GetUseFastMu300nsTrick() const { return  FastMu300nsTrick; }
inline G4PhysicsTable* Local_DsG4ScintSimple::GetPPOCDFTable() const { return thePPOCDFTable; }
inline G4PhysicsTable* Local_DsG4ScintSimple::GetBisCDFTable() const { return theBisCDFTable; }

#ifndef NDEBUG
inline void Local_DsG4ScintSimple::DumpPhysicsTable() const
{
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

#endif /* Local_DsG4ScintSimple_h */