// ---------------------------------------------------------------------------
// Local_DsOpticalModelHelper.hh
// Shared helper utilities for the JUNO New Optical Model PLUS.
//
// Consolidates code shared between Local_DsG4ScintSimple and Local_DsG4OpAbsReemit:
//   1. Combined direction + polarization sampling (reduced RNG calls)
//   2. CDF construction via trapezoidal integration
//   3. Optical photon secondary track creation
//   4. Mandatory material-property lookup
//   5. Micro-process tag registry (SetModelTag / GetModelTag)
//   6. Shared CDF tables (avoid duplicate memory across processes)
// ---------------------------------------------------------------------------
// Authors: Xuhui Yang, Blyth Simon
// ---------------------------------------------------------------------------

#ifndef Local_DsOpticalModelHelper_h
#define Local_DsOpticalModelHelper_h 1

#include "globals.hh"
#include "Randomize.hh"
#include "G4ThreeVector.hh"
#include "G4OpticalPhoton.hh"
#include "G4DynamicParticle.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4PhysicsOrderedFreeVector.hh"
#include "G4PhysicsTable.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4SystemOfUnits.hh"
#include "G4Version.hh"

#include <vector>

namespace DsOpticalModelHelper {

// ---------------------------------------------------------------------------
// Micro-process tagging registry.
// Names are registered in a fixed order in InitializeRegistry(), giving
// deterministic integer indices independent of construction order.
// Each process looks up its index in the constructor; the hot path uses
// the integer index directly via SetModelTag (no string operations).
// ---------------------------------------------------------------------------
inline std::vector<G4String>& GetProcNameRegistry() {
    static std::vector<G4String> registry;
    return registry;
}

inline void InitializeRegistry() {
    auto& registry = GetProcNameRegistry();
    if (!registry.empty()) return;

    // Scintillation components (Local_DsG4ScintSimple)
    registry.push_back("Scint_PPO_Fast");        // 0
    registry.push_back("Scint_PPO_Slow");        // 1
    registry.push_back("Scint_Bis_Fast");        // 2
    registry.push_back("Scint_Bis_Slow");        // 3

    // Absorption/Re-emission components (Local_DsG4OpAbsReemit)
    registry.push_back("AbsRe_LAB_NR_PPO");      // 4
    registry.push_back("AbsRe_LAB_NR_bisMSB");   // 5
    registry.push_back("AbsRe_PPO");             // 6
    registry.push_back("AbsRe_bisMSB");          // 7
}

inline G4int GetMicroProcessIndex(const G4String& name) {
    auto& registry = GetProcNameRegistry();
    for (size_t i = 0; i < registry.size(); ++i) {
        if (registry[i] == name) return static_cast<G4int>(i);
    }
    G4Exception("DsOpticalModelHelper::GetMicroProcessIndex",
                "NameNotRegistered", FatalException,
                ("Micro-process name not registered: " + name).c_str());
    return -1;
}

// ---------------------------------------------------------------------------
// Shared CDF vectors for PPO and bis-MSB emission spectra.
// Built once and reused across processes (~2 KB total).
// ---------------------------------------------------------------------------
inline G4PhysicsOrderedFreeVector*& GetSharedPPOCDF() {
    static G4PhysicsOrderedFreeVector* cdf = nullptr;
    return cdf;
}
inline G4PhysicsOrderedFreeVector*& GetSharedBisCDF() {
    static G4PhysicsOrderedFreeVector* cdf = nullptr;
    return cdf;
}

inline void BuildCDFVector(G4MaterialPropertyVector* srcVec,
                           G4PhysicsOrderedFreeVector* destVec);

// ---------------------------------------------------------------------------
// DestroySharedCDFTable — safely destroy a G4PhysicsTable that shares a
// single non-owning CDF entry at lsMaterialIndex.  Nulls out the shared
// entry before clearAndDestroy so the singleton is not double-deleted.
// ---------------------------------------------------------------------------
inline void DestroySharedCDFTable(G4PhysicsTable*& table, G4int lsMaterialIndex) {
    if (table) {
        if (lsMaterialIndex >= 0) {
            (*table)[lsMaterialIndex] = nullptr;
        }
        table->clearAndDestroy();
        delete table;
        table = nullptr;
    }
}

inline void BuildSharedCDFs(G4MaterialPropertiesTable* mpt) {
    if (GetSharedPPOCDF() != nullptr) return;

    G4PhysicsOrderedFreeVector* ppoCDF = new G4PhysicsOrderedFreeVector();
    G4PhysicsOrderedFreeVector* bisCDF = new G4PhysicsOrderedFreeVector();
    BuildCDFVector(mpt->GetProperty("PPOCOMPONENT"), ppoCDF);
    BuildCDFVector(mpt->GetProperty("bisMSBCOMPONENT"), bisCDF);
    GetSharedPPOCDF() = ppoCDF;
    GetSharedBisCDF() = bisCDF;
}

// ---------------------------------------------------------------------------
// SetModelTag / GetModelTag
// Store a lightweight integer index on a secondary track.
// G4>=11.0: SetCreatorModelID; G4<11.0: SetCreatorModelIndex.
// The getter GetCreatorModelID() exists in both versions.
// ---------------------------------------------------------------------------
inline void SetModelTag(G4Track* track, G4int tag) {
#if defined(G4VERSION_NUMBER) && G4VERSION_NUMBER >= 1100
    track->SetCreatorModelID(tag);
#else
// Note: In Geant4 10.x, GetCreatorModelID() returns the fCreatorModelIndex
// set by SetCreatorModelIndex (this confusing API naming is corrected in
// the Geant4 11.x series).
    track->SetCreatorModelIndex(tag);
#endif
}

inline G4int GetModelTag(const G4Track* track) {
    return track->GetCreatorModelID();
}

inline G4String GetMicroProcessName(const G4Track* track) {
    G4int tag = GetModelTag(track);
    if (tag >= 0) {
        auto& registry = GetProcNameRegistry();
        if (tag < static_cast<G4int>(registry.size())) {
            return registry[tag];
        }
    }
    return "";
}

// ---------------------------------------------------------------------------
// GetConstPropertyMandatory — fetch a constant material property, with fatal
// error if the property is missing or the table is null.
// ---------------------------------------------------------------------------
inline G4double GetConstPropertyMandatory(G4MaterialPropertiesTable* mpt,
                                          const G4String& key) {
    if (!mpt || !mpt->ConstPropertyExists(key.c_str())) {
        G4Exception("DsOpticalModelHelper::GetConstPropertyMandatory",
                    "MissingProperty", FatalException,
                    ("Mandatory LS optical property not found: " + key).c_str());
        return 0.0;
    }
    return mpt->GetConstProperty(key.c_str());
}

// ---------------------------------------------------------------------------
// SampleDirectionAndPolarization — combined isotropic direction + random
// polarization.  Uses Marsaglia polar method (avoids sin/cos) and builds
// an orthonormal basis perpendicular to the sampled direction.
// ~3.5 RNG calls per photon.
// ---------------------------------------------------------------------------
inline void SampleDirectionAndPolarization(G4ThreeVector& direction,
                                           G4ThreeVector& polarization) {
    G4double u, v, S, fac;
    do {
        u = 2.0 * G4UniformRand() - 1.0;
        v = 2.0 * G4UniformRand() - 1.0;
        S = u * u + v * v;
    } while (S >= 1.0 || S == 0.0);
    fac = 2.0 * std::sqrt(1.0 - S);
    direction = G4ThreeVector(u * fac, v * fac, 1.0 - 2.0 * S);

    // Build a right-handed orthonormal basis in the plane perpendicular to direction.
    // Use the smallest component to avoid numerical degeneracy in the cross product.
    // Since |direction| = 1, the norm of the cross-product vector is sqrt(1 - min_dir^2),
    // computed directly to avoid a temporary G4ThreeVector and an extra sqrt.
    G4double ax = std::abs(direction.x());
    G4double ay = std::abs(direction.y());
    G4double az = std::abs(direction.z());
    G4ThreeVector perp;
    if (ax <= ay && ax <= az) {
        G4double norm = std::sqrt(1.0 - direction.x()*direction.x());
        perp = G4ThreeVector(0.0, direction.z()/norm, -direction.y()/norm);
    } else if (ay <= ax && ay <= az) {
        G4double norm = std::sqrt(1.0 - direction.y()*direction.y());
        perp = G4ThreeVector(-direction.z()/norm, 0.0, direction.x()/norm);
    } else {
        G4double norm = std::sqrt(1.0 - direction.z()*direction.z());
        perp = G4ThreeVector(-direction.y()/norm, direction.x()/norm, 0.0);
    }

    G4ThreeVector perp2 = direction.cross(perp);

    G4double phi = 2.0 * M_PI * G4UniformRand();
    G4double cosPhi = std::cos(phi);
    G4double sinPhi = std::sin(phi);
    polarization = cosPhi * perp + sinPhi * perp2;
}

// ---------------------------------------------------------------------------
// BuildCDFVector — trapezoidal integration of a PDF vector to produce a CDF
// for inverse-transform sampling.  Skips vectors with negative first intensity.
// ---------------------------------------------------------------------------
inline void BuildCDFVector(G4MaterialPropertyVector* srcVec,
                           G4PhysicsOrderedFreeVector* destVec) {
    if (!srcVec || srcVec->GetVectorLength() == 0) return;

    G4double currentIN = (*srcVec)[0];
    if (currentIN < 0.0) return;

    G4double currentPM = srcVec->Energy(0);
    G4double currentCII = 0.0;
    destVec->InsertValues(currentPM, currentCII);

    G4double prevPM  = currentPM;
    G4double prevCII = currentCII;
    G4double prevIN  = currentIN;

    for (size_t ii = 1; ii < srcVec->GetVectorLength(); ++ii) {
        currentPM = srcVec->Energy(ii);
        currentIN = (*srcVec)[ii];
        currentCII = prevCII + 0.5 * (prevIN + currentIN) * (currentPM - prevPM);
        destVec->InsertValues(currentPM, currentCII);
        prevPM  = currentPM;
        prevCII = currentCII;
        prevIN  = currentIN;
    }
}

// ---------------------------------------------------------------------------
// CreateOpticalPhotonTrack — create a secondary optical-photon track with
// the given kinetic energy, momentum, polarization, position, time, and
// touchable handle.
// ---------------------------------------------------------------------------
inline G4Track* CreateOpticalPhotonTrack(G4double energy,
                                         const G4ThreeVector& momentum,
                                         const G4ThreeVector& polarization,
                                         const G4ThreeVector& position,
                                         G4double time,
                                         const G4TouchableHandle& touchable,
                                         G4int parentID,
                                         G4double weight = 1.0) {
    auto* photon = new G4DynamicParticle(G4OpticalPhoton::OpticalPhoton(),
                                         momentum);
    photon->SetPolarization(polarization.x(),
                            polarization.y(),
                            polarization.z());
    photon->SetKineticEnergy(energy);

    auto* track = new G4Track(photon, time, position);
    track->SetWeight(weight);
    track->SetTouchableHandle(touchable);
    track->SetParentID(parentID);
    return track;
}

}  // namespace DsOpticalModelHelper

#endif  // Local_DsOpticalModelHelper_h