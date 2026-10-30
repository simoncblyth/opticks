#include "Local_DsG4ScintSimple.h"
#include "G4UnitsTable.hh"
#include "G4LossTableManager.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "globals.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4OpticalPhoton.hh"
#include "G4Poisson.hh"
#include "Local_DsOpticalModelHelper.hh"

#include "U4.hh"

// Micro-process names for the four scintillation components.
// Looked up in the constructor; used as integer indices in the hot path.
namespace {
    constexpr std::array<const char*, Local_DsG4ScintSimple::kNumComponents> kProcName = {{
        "Scint_PPO_Fast", "Scint_PPO_Slow",
        "Scint_Bis_Fast", "Scint_Bis_Slow"
    }};
}

Local_DsG4ScintSimple::Local_DsG4ScintSimple(const G4String& processName, G4ProcessType type)
    : G4VRestDiscreteProcess(processName, type)
    , thePPOCDFTable(nullptr)
    , theBisCDFTable(nullptr)
    , fEnableQuenching(true)
    , birks1{}, birks2{}
    , flagDecayTimeFast(true), flagDecayTimeSlow(true)
    , fTrackSecondariesFirst(false)
    , YieldFactor(1.0)
    , FastMu300nsTrick(false)
    , fPhotonWeight(1.0)
    , fApplyPreQE(false)
    , fPreQE(1.)
    , m_noop(false)
    , m_rem_ppo(0.0), m_rem_bis(0.0)
    , m_rem_ppo_fast(0.0), m_rem_bis_fast(0.0)
    , m_ls_material_index(-1)
{
    SetProcessSubType(fScintillation);

    DsOpticalModelHelper::InitializeRegistry();
    for (G4int i = 0; i < kNumComponents; ++i) {
        m_procIndex[i] = DsOpticalModelHelper::GetMicroProcessIndex(kProcName[i]);
    }

    if (verboseLevel > 0) {
        G4cout << GetProcessName() << " is created " << G4endl;
    }

    BuildThePhysicsTable();
}

Local_DsG4ScintSimple::~Local_DsG4ScintSimple()
{
    DsOpticalModelHelper::DestroySharedCDFTable(thePPOCDFTable, m_ls_material_index);
    DsOpticalModelHelper::DestroySharedCDFTable(theBisCDFTable, m_ls_material_index);
}

G4VParticleChange* Local_DsG4ScintSimple::AtRestDoIt(const G4Track& aTrack, const G4Step& aStep)
{
    return Local_DsG4ScintSimple::PostStepDoIt(aTrack, aStep);
}

G4VParticleChange* Local_DsG4ScintSimple::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
{
    aParticleChange.Initialize(aTrack);

    if (m_noop) {
        aParticleChange.SetNumberOfSecondaries(0);
        return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    G4double TotalEnergyDeposit = aStep.GetTotalEnergyDeposit();
    if (TotalEnergyDeposit <= 0.0) {
        return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    const G4Material* aMaterial = aTrack.GetMaterial();
    if (static_cast<G4int>(aMaterial->GetIndex()) != m_ls_material_index) {
        return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
    G4int pdgCode = aParticle->GetDefinition()->GetPDGEncoding();

    G4StepPoint* pPreStepPoint  = aStep.GetPreStepPoint();
    G4StepPoint* pPostStepPoint = aStep.GetPostStepPoint();

    G4ThreeVector x0 = pPreStepPoint->GetPosition();
    G4double      t0 = pPreStepPoint->GetGlobalTime();

    G4int NumTracks = 0;
    G4double weight = 1.0;

    // -----------------------------------------------------------------------
    // Birks' law:  dE/dx quenching factor = 1/(1 + k1*delta + k2*delta^2)
    // where delta = (dE/dx)/density.
    // -----------------------------------------------------------------------
    G4double ScintillationYield = m_scintillation_yield;
    G4double ResolutionScale = m_resolution_scale;

    G4double dE = TotalEnergyDeposit;
    G4double dx = aStep.GetStepLength();
    G4double dE_dx = dE/dx;

    if (pdgCode == 22 && dE > 0) {
        G4LossTableManager* manager = G4LossTableManager::Instance();
        dE_dx = dE / manager->GetRange(G4Electron::Electron(), dE, aTrack.GetMaterialCutsCouple());
    }

    G4double birkDelta = dE_dx / aMaterial->GetDensity();

    // Map PDG code to Birks constants and bis-MSB particle type via compact lookup table
    struct PDGMapEntry {
        G4int pdg;
        G4int birksIdx;
        G4int bisType;
    };
    static constexpr PDGMapEntry PDG_MAP[] = {
        {  22,        static_cast<G4int>(BirksIdx::kEe), static_cast<G4int>(BisParticle::kGamma)    },
        {  11,        static_cast<G4int>(BirksIdx::kEe), static_cast<G4int>(BisParticle::kElectron) },
        { -11,        static_cast<G4int>(BirksIdx::kEe), static_cast<G4int>(BisParticle::kElectron) },
        { 2212,       static_cast<G4int>(BirksIdx::kP),  static_cast<G4int>(BisParticle::kProton)   },
        { 2112,       static_cast<G4int>(BirksIdx::kP),  static_cast<G4int>(BisParticle::kNeutron)  },
        { 1000020040, static_cast<G4int>(BirksIdx::kA),  static_cast<G4int>(BisParticle::kAlpha)    },
    };
    constexpr G4int NUM_PDG_MAP = sizeof(PDG_MAP) / sizeof(PDG_MAP[0]);

    G4int birksIdx = static_cast<G4int>(BirksIdx::kEe);
    G4int bisType  = static_cast<G4int>(BisParticle::kElectron);
    G4bool foundPDG = false;

    for (G4int ii = 0; ii < NUM_PDG_MAP; ++ii) {
        if (PDG_MAP[ii].pdg == pdgCode) {
            birksIdx = PDG_MAP[ii].birksIdx;
            bisType  = PDG_MAP[ii].bisType;
            foundPDG = true;
            break;
        }
    }
    // Fallback for unknown PDG codes: classify by charge
    if (!foundPDG) {
        if (std::abs(aParticle->GetCharge()) < 1.5) {
            birksIdx = static_cast<G4int>(BirksIdx::kP);
            bisType  = static_cast<G4int>(BisParticle::kProton);
        } else {
            birksIdx = static_cast<G4int>(BirksIdx::kA);
            bisType  = static_cast<G4int>(BisParticle::kAlpha);
        }
        if (verboseLevel > 1) {
            G4cout << "Local_DsG4ScintSimple: Unrecognized PDG " << pdgCode
                   << " (charge=" << aParticle->GetCharge()
                   << "), treated as "
                   << (birksIdx == static_cast<G4int>(BirksIdx::kP) ? "proton-like" : "alpha-like")
                   << G4endl;
        }
    }

    G4double birk1 = birks1[birksIdx];
    G4double birk2 = birks2[birksIdx];

    G4double lab2bis_ratio   = m_ratio_bis[bisType];
    G4double lab2bis_tauslow = m_tauslow_bis[bisType];

    G4double QuenchedTotalEnergyDeposit = TotalEnergyDeposit;
    if (fEnableQuenching) {
        QuenchedTotalEnergyDeposit = TotalEnergyDeposit / (1 + birk1*birkDelta + birk2*birkDelta*birkDelta);
    }

    // FastMu300nsTrick: suppress scintillation before 300 ns for muon tracks
    if (FastMu300nsTrick && aStep.GetTrack()->GetGlobalTime()/ns <= 300.) {
        ScintillationYield = 0.;
    } else {
        ScintillationYield = YieldFactor * ScintillationYield;
    }

    G4double MeanNumberOfPhotons = ScintillationYield * QuenchedTotalEnergyDeposit;
    G4double MeanNumberOfTracks  = MeanNumberOfPhotons / fPhotonWeight;

    if (fApplyPreQE) MeanNumberOfTracks *= fPreQE;

    // Fluctuation: Gaussian for high yields, Poisson for low yields
    if (MeanNumberOfTracks > 10.) {
        G4double sigma = ResolutionScale * std::sqrt(MeanNumberOfTracks);
        NumTracks = static_cast<G4int>(G4RandGauss::shoot(MeanNumberOfTracks, sigma) + 0.5);
    } else {
        NumTracks = static_cast<G4int>(G4Poisson(MeanNumberOfTracks));
    }

    weight *= fPhotonWeight;

    if (NumTracks <= 0) {
        aParticleChange.SetNumberOfSecondaries(0);
        return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    // Error Diffusion: split between PPO and bis-MSB components
    G4double exact_ppo = NumTracks * m_prob_lab2ppo + m_rem_ppo;
    G4int NumPPO = static_cast<G4int>(exact_ppo);
    m_rem_ppo = exact_ppo - NumPPO;

    G4double exact_bis = NumTracks * m_prob_lab2bis + m_rem_bis;
    G4int NumBis = static_cast<G4int>(exact_bis);
    m_rem_bis = exact_bis - NumBis;

    NumTracks = NumPPO + NumBis;

    if (NumTracks <= 0) {
        aParticleChange.SetNumberOfSecondaries(0);
        return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    aParticleChange.SetNumberOfSecondaries(NumTracks);

    if (fTrackSecondariesFirst) {
        if (aTrack.GetTrackStatus() == fAlive)
            aParticleChange.ProposeTrackStatus(fSuspend);
    }

    // Error Diffusion: split each component into Fast (Singlet) / Slow (Triplet)
    G4double exact_ppo_fast = NumPPO * m_ratio_ppo + m_rem_ppo_fast;
    G4int numPPO_fast = static_cast<G4int>(exact_ppo_fast);
    m_rem_ppo_fast = exact_ppo_fast - numPPO_fast;
    G4int numPPO_slow = NumPPO - numPPO_fast;

    G4double exact_bis_fast = NumBis * lab2bis_ratio + m_rem_bis_fast;
    G4int numBis_fast = static_cast<G4int>(exact_bis_fast);
    m_rem_bis_fast = exact_bis_fast - numBis_fast;
    G4int numBis_slow = NumBis - numBis_fast;

    // Structured arrays for the 4-component loop: [0] PPO Fast, [1] PPO Slow, [2] Bis Fast, [3] Bis Slow
    const std::array<G4int, kNumComponents> comp_photons = {{numPPO_fast, numPPO_slow, numBis_fast, numBis_slow}};
    const std::array<G4double, kNumComponents> comp_taus_nr = {{
        flagDecayTimeFast ? m_tau_lab2ppo : 0.0,
        flagDecayTimeSlow ? m_tau_ppo_slow : 0.0,
        flagDecayTimeFast ? m_tau_lab2bis : 0.0,
        flagDecayTimeSlow ? lab2bis_tauslow : 0.0
    }};

    // Pre-fetch the two shared CDF pointers (4 table lookups reduced to 2)
    auto* ppoCDF = static_cast<G4PhysicsOrderedFreeVector*>((*thePPOCDFTable)(m_ls_material_index));
    auto* bisCDF = static_cast<G4PhysicsOrderedFreeVector*>((*theBisCDFTable)(m_ls_material_index));
    const std::array<G4PhysicsOrderedFreeVector*, kNumComponents> comp_integral = {{ppoCDF, ppoCDF, bisCDF, bisCDF}};

    // Cache loop-invariant values
    G4double stepLength    = aStep.GetStepLength();
    G4double avgVelocity   = (pPreStepPoint->GetVelocity() + pPostStepPoint->GetVelocity()) * 0.5;
    G4ThreeVector deltaPos = aStep.GetDeltaPosition();
    G4TouchableHandle touchable = aStep.GetPreStepPoint()->GetTouchableHandle();
    G4int parentID = aTrack.GetTrackID();
    G4double charge = aParticle->GetDefinition()->GetPDGCharge();
    bool isCharged = (charge != 0);

    aParticleChange.SetSecondaryWeightByProcess(true);

    //--- opticks::U4 ---//
    if( m_opticksMode == 2 || m_opticksMode == 3 ) U4::GenPhotonAncestor(&aTrack);
    //-------------------//

    // Photon Generation Loop
    for (G4int k = 0; k < kNumComponents; ++k) {
        G4int n_base = comp_photons[k];
        if (n_base <= 0) continue;

        G4double tau_nr  = comp_taus_nr[k];
        G4int procIndex = m_procIndex[k];
        G4PhysicsOrderedFreeVector* ScintillationIntegral = comp_integral[k];
        G4bool hasNR = (tau_nr > 0.0);

        G4double CIImax = ScintillationIntegral->GetMaxValue();

        //--- opticks::U4 ---//
        if( m_opticksMode == 1 || m_opticksMode == 3 ) U4::CollectGenstep_DsG4ScintSimple(&aTrack, &aStep, n_base, k, tau_nr);
        //-------------------//

        if( m_opticksMode == 1 ) continue;    // GPU-only: no CPU secondaries

        for (G4int i = 0; i < n_base; i++) {
            //--- opticks::U4 ---//
            if( m_opticksMode == 2 || m_opticksMode == 3 ) U4::GenPhotonBegin(i);
            //-------------------//

            // 1. Inverse Transform Sampling for emission energy
            G4double CIIvalue = G4UniformRand() * CIImax;
            G4double sampledEnergy = ScintillationIntegral->GetEnergy(CIIvalue);

            // 2. Isotropic momentum & polarization (combined Marsaglia call)
            G4ThreeVector photonMomentum;
            G4ThreeVector photonPolarization;
            DsOpticalModelHelper::SampleDirectionAndPolarization(photonMomentum, photonPolarization);

            // 3. Time and position sampling
            G4double rand = isCharged ? G4UniformRand() : 1.0;
            G4double ds = rand * stepLength;
            G4double deltaTime = ds / avgVelocity;

            if (hasNR) {
                deltaTime += -std::log(G4UniformRand()) * tau_nr;
            }

            G4ThreeVector aSecondaryPosition = x0 + rand * deltaPos;
            G4double aSecondaryTime = t0 + deltaTime;

            // 4. Create secondary photon track
            G4Track* aSecondaryTrack = DsOpticalModelHelper::CreateOpticalPhotonTrack(
                sampledEnergy, photonMomentum, photonPolarization,
                aSecondaryPosition, aSecondaryTime,
                touchable, parentID, weight);

            DsOpticalModelHelper::SetModelTag(aSecondaryTrack, procIndex);

            aParticleChange.AddSecondary(aSecondaryTrack);

            //--- opticks::U4 ---//
            if( m_opticksMode == 2 || m_opticksMode == 3 ) U4::GenPhotonEnd(i, aSecondaryTrack);
            //-------------------//
        }
    }

    return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
}

void Local_DsG4ScintSimple::BuildThePhysicsTable()
{
    if (thePPOCDFTable && theBisCDFTable) return;

    const G4MaterialTable* theMaterialTable = G4Material::GetMaterialTable();
    G4int numOfMaterials = G4Material::GetNumberOfMaterials();

    if (!thePPOCDFTable) thePPOCDFTable = new G4PhysicsTable(numOfMaterials);
    if (!theBisCDFTable) theBisCDFTable = new G4PhysicsTable(numOfMaterials);

    if (verboseLevel > 1) {
        G4cout << "Local_DsG4ScintSimple: Building physics tables for "
               << numOfMaterials << " materials" << G4endl;
    }

    for (G4int i = 0; i < numOfMaterials; i++) {
        G4PhysicsOrderedFreeVector* ppoCDF = nullptr;
        G4PhysicsOrderedFreeVector* bisCDF = nullptr;

        G4Material* aMaterial = (*theMaterialTable)[i];
        G4MaterialPropertiesTable* mpt = aMaterial->GetMaterialPropertiesTable();

        if (mpt) {
            // Detect LS material by the presence of LAB2PPO_PROB
            if (m_ls_material_index < 0 && mpt->ConstPropertyExists("LAB2PPO_PROB")) {
                m_ls_material_index = i;

                using DsOpticalModelHelper::GetConstPropertyMandatory;

                m_scintillation_yield = GetConstPropertyMandatory(mpt, "SCINTILLATIONYIELD");
                const G4MaterialPropertyVector* ptable = mpt->GetProperty("RESOLUTIONSCALE");
                m_resolution_scale = ptable ? ptable->Value(0) : 1.0;

                m_prob_lab2ppo = GetConstPropertyMandatory(mpt, "LAB2PPO_PROB");
                m_prob_lab2bis = GetConstPropertyMandatory(mpt, "LAB2BIS_PROB");
                m_tau_lab2ppo  = GetConstPropertyMandatory(mpt, "LAB2PPO_TAU");
                m_tau_lab2bis  = GetConstPropertyMandatory(mpt, "LAB2BIS_TAU");

                m_ratio_ppo    = GetConstPropertyMandatory(mpt, "LAB2PPO_RATIO");
                m_tau_ppo_slow = GetConstPropertyMandatory(mpt, "LAB2PPO_TAUSLOW");

                m_ratio_bis[static_cast<G4int>(BisParticle::kElectron)] =
                    GetConstPropertyMandatory(mpt, "LAB2BIS_ELECTRONRATIO");
                m_tauslow_bis[static_cast<G4int>(BisParticle::kElectron)] =
                    GetConstPropertyMandatory(mpt, "LAB2BIS_ELECTRONTAUSLOW");
                m_ratio_bis[static_cast<G4int>(BisParticle::kAlpha)] =
                    GetConstPropertyMandatory(mpt, "LAB2BIS_ALPHARATIO");
                m_tauslow_bis[static_cast<G4int>(BisParticle::kAlpha)] =
                    GetConstPropertyMandatory(mpt, "LAB2BIS_ALPHATAUSLOW");
                m_ratio_bis[static_cast<G4int>(BisParticle::kGamma)] =
                    GetConstPropertyMandatory(mpt, "LAB2BIS_GAMMARATIO");
                m_tauslow_bis[static_cast<G4int>(BisParticle::kGamma)] =
                    GetConstPropertyMandatory(mpt, "LAB2BIS_GAMMATAUSLOW");
                m_ratio_bis[static_cast<G4int>(BisParticle::kProton)] =
                    GetConstPropertyMandatory(mpt, "LAB2BIS_PROTONRATIO");
                m_tauslow_bis[static_cast<G4int>(BisParticle::kProton)] =
                    GetConstPropertyMandatory(mpt, "LAB2BIS_PROTONTAUSLOW");
                m_ratio_bis[static_cast<G4int>(BisParticle::kNeutron)] =
                    GetConstPropertyMandatory(mpt, "LAB2BIS_NEUTRONRATIO");
                m_tauslow_bis[static_cast<G4int>(BisParticle::kNeutron)] =
                    GetConstPropertyMandatory(mpt, "LAB2BIS_NEUTRONTAUSLOW");

                m_tau_intrinsic_ppo = GetConstPropertyMandatory(mpt, "PPO_INTRINSICTAU");
                m_tau_intrinsic_bis = GetConstPropertyMandatory(mpt, "BISMSB_INTRINSICTAU");

                G4cout << "Local_DsG4ScintSimple: Successfully Cached JUNO New Optical Model Parameters." << G4endl;
            }

            // Only the LS material has PPO/bisMSB data; non-LS entries store nullptr.
            // Uses shared CDFs from DsOpticalModelHelper to avoid duplication.
            if (i == m_ls_material_index) {
                DsOpticalModelHelper::BuildSharedCDFs(mpt);
                ppoCDF = DsOpticalModelHelper::GetSharedPPOCDF();
                bisCDF = DsOpticalModelHelper::GetSharedBisCDF();
            }
        }

        thePPOCDFTable->insertAt(i, ppoCDF);
        theBisCDFTable->insertAt(i, bisCDF);
    }

    if (m_ls_material_index < 0) {
        G4Exception("Local_DsG4ScintSimple::BuildThePhysicsTable",
                    "LSNotFound", JustWarning,
                    "No LS material found (LAB2PPO_PROB not present). "
                    "Scintillation will be disabled.");
    }
}

G4double Local_DsG4ScintSimple::GetMeanFreePath(const G4Track&, G4double, G4ForceCondition* condition)
{
    *condition = StronglyForced;
    return DBL_MAX;
}

G4double Local_DsG4ScintSimple::GetMeanLifeTime(const G4Track&, G4ForceCondition* condition)
{
    *condition = Forced;
    return DBL_MAX;
}

G4bool Local_DsG4ScintSimple::IsApplicable(const G4ParticleDefinition& aParticleType)
{
    // ScintSimple generates primary scintillation only — reject optical photons
    return (&aParticleType != G4OpticalPhoton::OpticalPhoton());
}