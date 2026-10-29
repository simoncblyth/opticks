#include "G4ios.hh"
#include "G4OpProcessSubType.hh"
#include "G4PhysicalConstants.hh"
#include "Local_DsG4OpAbsReemit.h"
#include "Local_DsOpticalModelHelper.hh"

#include "U4.hh"

Local_DsG4OpAbsReemit::Local_DsG4OpAbsReemit(ReemitComponent comp, const G4String& processName, G4ProcessType type)
  : G4VDiscreteProcess(processName, type)
  , thePPOCDFTable(nullptr)
  , theBisCDFTable(nullptr)
  , m_comp(comp)
  , m_ls_material_index(-1)
  , m_prob_lab2ppo(1.0), m_prob_lab2bis(0.0)
  , m_tau_lab2ppo(0.0),  m_tau_lab2bis(0.0)
  , m_tau_intrinsic_ppo(0.0), m_tau_intrinsic_bis(0.0)
  , m_reemiprob_ppo_vec(nullptr), m_reemiprob_bis_vec(nullptr)
{
    DsOpticalModelHelper::InitializeRegistry();
    m_procIndex_PPO_LAB = DsOpticalModelHelper::GetMicroProcessIndex("AbsRe_LAB_NR_PPO");
    m_procIndex_Bis_LAB = DsOpticalModelHelper::GetMicroProcessIndex("AbsRe_LAB_NR_bisMSB");
    m_procIndex_PPO     = DsOpticalModelHelper::GetMicroProcessIndex("AbsRe_PPO");
    m_procIndex_Bis     = DsOpticalModelHelper::GetMicroProcessIndex("AbsRe_bisMSB");

    switch (m_comp) {
        case ReemitComponent::kSolvent:
            SetProcessSubType(fOpAbsorption);
            m_label_abslen = "ABSLENGTH";
            break;
        case ReemitComponent::kPPO:
            SetProcessSubType(fOpWLS);
            m_label_abslen = "PPOABSLENGTH";
            break;
        case ReemitComponent::kbisMSB:
            SetProcessSubType(fOpWLS);
            m_label_abslen = "bisMSBABSLENGTH";
            break;
    }

    if (verboseLevel > 0) {
        G4cout << GetProcessName() << " is created (comp="
               << static_cast<G4int>(m_comp) << ")" << G4endl;
    }

    BuildThePhysicsTable();
}

Local_DsG4OpAbsReemit::~Local_DsG4OpAbsReemit()
{
    DsOpticalModelHelper::DestroySharedCDFTable(thePPOCDFTable, m_ls_material_index);
    DsOpticalModelHelper::DestroySharedCDFTable(theBisCDFTable, m_ls_material_index);
}

G4VParticleChange*
Local_DsG4OpAbsReemit::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
{
    aParticleChange.Initialize(aTrack);
    aParticleChange.ProposeTrackStatus(fStopAndKill);

    const G4Material* aMaterial = aTrack.GetMaterial();
    G4int materialIndex = aMaterial->GetIndex();
    G4MaterialPropertiesTable* mpt = aMaterial->GetMaterialPropertiesTable();

    if (!mpt) {
        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    G4double tau_nr  = 0.0;
    G4double tau_rad = 0.0;
    G4int procIndex = -1;
    G4MaterialPropertyVector* target_reemiprob_vec = nullptr;
    G4PhysicsOrderedFreeVector* WLSIntegral = nullptr;

    if (m_comp == ReemitComponent::kSolvent) {
        // Non-LS solvents (steel, water) act as pure absorption
        if (materialIndex != m_ls_material_index) {
            return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
        }

        G4double rand_fate = G4UniformRand();
        if (rand_fate < m_prob_lab2ppo) {
            target_reemiprob_vec = m_reemiprob_ppo_vec;
            WLSIntegral = static_cast<G4PhysicsOrderedFreeVector*>((*thePPOCDFTable)(materialIndex));
            tau_nr  = m_tau_lab2ppo;
            tau_rad = m_tau_intrinsic_ppo;
            procIndex = m_procIndex_PPO_LAB;
        } else if (rand_fate < m_prob_lab2ppo + m_prob_lab2bis) {
            target_reemiprob_vec = m_reemiprob_bis_vec;
            WLSIntegral = static_cast<G4PhysicsOrderedFreeVector*>((*theBisCDFTable)(materialIndex));
            tau_nr  = m_tau_lab2bis;
            tau_rad = m_tau_intrinsic_bis;
            procIndex = m_procIndex_Bis_LAB;
        } else {
            return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
        }
    } else {
        if (materialIndex == m_ls_material_index) {
            target_reemiprob_vec = (m_comp == ReemitComponent::kPPO) ?
                                    m_reemiprob_ppo_vec : m_reemiprob_bis_vec;
        }
        if (m_comp == ReemitComponent::kPPO) {
            WLSIntegral = static_cast<G4PhysicsOrderedFreeVector*>((*thePPOCDFTable)(materialIndex));
            tau_rad = m_tau_intrinsic_ppo;
            procIndex = m_procIndex_PPO;
        } else if (m_comp == ReemitComponent::kbisMSB) {
            WLSIntegral = static_cast<G4PhysicsOrderedFreeVector*>((*theBisCDFTable)(materialIndex));
            tau_rad = m_tau_intrinsic_bis;
            procIndex = m_procIndex_Bis;
        }
    }

    // Re-emission probability check
    if (!target_reemiprob_vec) {
        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }
    G4double p_reemission = target_reemiprob_vec->Value(aTrack.GetKineticEnergy());
    if (G4UniformRand() >= p_reemission) {
        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    // Inverse Transform Sampling using CDF
    G4double sampledEnergy = 0.;
    if (WLSIntegral && WLSIntegral->GetVectorLength() > 0) {
        G4double CIImax = WLSIntegral->GetMaxValue();
        G4double CIIvalue = G4UniformRand() * CIImax;
        sampledEnergy = WLSIntegral->GetEnergy(CIIvalue);
    } else {
        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    // Time delay: NR transfer + intrinsic emission
    G4double deltaTime = 0.;
    if (tau_nr > 0.0)  deltaTime += -std::log(G4UniformRand()) * tau_nr;
    if (tau_rad > 0.0) deltaTime += -std::log(G4UniformRand()) * tau_rad;

    G4ThreeVector photonMomentum;
    G4ThreeVector photonPolarization;
    DsOpticalModelHelper::SampleDirectionAndPolarization(photonMomentum, photonPolarization);

    G4StepPoint* pPostStepPoint = aStep.GetPostStepPoint();
    G4double aSecondaryTime = pPostStepPoint->GetGlobalTime() + deltaTime;
    G4ThreeVector aSecondaryPosition = pPostStepPoint->GetPosition();

    //--- opticks::U4 ---//
    if( m_opticksMode == 2 || m_opticksMode == 3 )
    {
        U4::GenPhotonAncestor(&aTrack);
        U4::GenPhotonBegin(0);
    }
    //-------------------//

    G4Track* aSecondaryTrack = DsOpticalModelHelper::CreateOpticalPhotonTrack(
        sampledEnergy, photonMomentum, photonPolarization,
        aSecondaryPosition, aSecondaryTime,
        aTrack.GetTouchableHandle(), aTrack.GetTrackID());

    DsOpticalModelHelper::SetModelTag(aSecondaryTrack, procIndex);

    aParticleChange.SetNumberOfSecondaries(1);
    aParticleChange.AddSecondary(aSecondaryTrack);

    //--- opticks::U4 ---//
    if( m_opticksMode == 2 || m_opticksMode == 3 ) U4::GenPhotonEnd(0, aSecondaryTrack);
    //-------------------//

    return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
}

G4double Local_DsG4OpAbsReemit::GetMeanFreePath(const G4Track& aTrack, G4double, G4ForceCondition*)
{
    const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
    const G4Material* aMaterial = aTrack.GetMaterial();

    G4double thePhotonEnergy = aParticle->GetTotalEnergy();
    G4MaterialPropertiesTable* mpt = aMaterial->GetMaterialPropertiesTable();
    G4double AttenuationLength = DBL_MAX;

    if (mpt) {
        // label dispatch for kSolvent: on the LS material the LAB solvent
        // absorption is registered as "LABABSLENGTH" (see
        // LSExpDetectorConstructionMaterial.icc); other materials keep "ABSLENGTH".
        G4String label_abslen = m_label_abslen;
        if (m_comp == ReemitComponent::kSolvent &&
            G4int(aMaterial->GetIndex()) == m_ls_material_index) {
            label_abslen = "LABABSLENGTH";
        }
        G4MaterialPropertyVector* v = mpt->GetProperty(label_abslen.c_str());
        if (v) {
            AttenuationLength = v->Value(thePhotonEnergy);
        }
    }

    return AttenuationLength;
}

void Local_DsG4OpAbsReemit::BuildThePhysicsTable()
{
    if (thePPOCDFTable || theBisCDFTable) return;

    const G4MaterialTable* theMaterialTable = G4Material::GetMaterialTable();
    G4int numOfMaterials = G4Material::GetNumberOfMaterials();

    thePPOCDFTable = new G4PhysicsTable(numOfMaterials);
    theBisCDFTable = new G4PhysicsTable(numOfMaterials);

    if (verboseLevel > 1) {
        G4cout << "Local_DsG4OpAbsReemit [" << GetProcessName()
               << "]: Building physics tables for "
               << numOfMaterials << " materials" << G4endl;
    }

    for (G4int i = 0; i < numOfMaterials; i++) {
        G4Material* aMaterial = (*theMaterialTable)[i];
        G4MaterialPropertiesTable* mpt = aMaterial->GetMaterialPropertiesTable();

        using DsOpticalModelHelper::GetConstPropertyMandatory;
        using DsOpticalModelHelper::BuildCDFVector;

        G4PhysicsOrderedFreeVector* ppoCDF = nullptr;
        G4PhysicsOrderedFreeVector* bisCDF = nullptr;

        if (mpt) {
            // Detect LS material: kSolvent uses LAB2PPO_PROB, kPPO/kbisMSB use PPO_INTRINSICTAU
            G4bool isLS = false;
            if (m_comp == ReemitComponent::kSolvent) {
                isLS = (m_ls_material_index < 0 && mpt->ConstPropertyExists("LAB2PPO_PROB"));
            } else {
                isLS = (m_ls_material_index < 0 && mpt->ConstPropertyExists("PPO_INTRINSICTAU"));
            }

            if (isLS) {
                m_ls_material_index = i;

                m_tau_intrinsic_ppo = GetConstPropertyMandatory(mpt, "PPO_INTRINSICTAU");
                m_tau_intrinsic_bis = GetConstPropertyMandatory(mpt, "BISMSB_INTRINSICTAU");

                m_reemiprob_ppo_vec = mpt->GetProperty("PPOREEMISSIONPROB");
                m_reemiprob_bis_vec = mpt->GetProperty("bisMSBREEMISSIONPROB");

                if (m_comp == ReemitComponent::kSolvent) {
                    m_prob_lab2ppo = GetConstPropertyMandatory(mpt, "LAB2PPO_PROB");
                    m_prob_lab2bis = GetConstPropertyMandatory(mpt, "LAB2BIS_PROB");
                    m_tau_lab2ppo  = GetConstPropertyMandatory(mpt, "LAB2PPO_TAU");
                    m_tau_lab2bis  = GetConstPropertyMandatory(mpt, "LAB2BIS_TAU");
                }

                DsOpticalModelHelper::BuildSharedCDFs(mpt);
                ppoCDF = DsOpticalModelHelper::GetSharedPPOCDF();
                bisCDF = DsOpticalModelHelper::GetSharedBisCDF();
            }
        }

        thePPOCDFTable->insertAt(i, ppoCDF);
        theBisCDFTable->insertAt(i, bisCDF);
    }

    if (m_ls_material_index < 0) {
        G4Exception("Local_DsG4OpAbsReemit::BuildThePhysicsTable",
                    "LSNotFound", JustWarning,
                    "No LS material found. Re-emission will be disabled.");
    }
}