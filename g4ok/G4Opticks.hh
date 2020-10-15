/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include "plog/Severity.h"
#include "G4ThreeVector.hh"

#include "G4OK_API_EXPORT.hh"

template <typename T> class NPY ; 
class NLookup ; 
class NPho ; 
class Opticks;
class OpticksEvent;
class OpMgr;
class GGeo ; 
class GBndLib ; 

class CTraverser ; 
class CMaterialTable ; 
class CGenstepCollector ; 
class CPrimaryCollector ; 
class CPhotonCollector ; 
class C4PhotonCollector ; 

class G4Run;
class G4Event; 
class G4Track; 
class G4Step; 
class G4VPhysicalVolume ;
class G4VParticleChange ; 
class G4PVPlacement ; 

#include "G4Types.hh"

/**
G4Opticks : interface for steering an Opticks instance embedded within a Geant4 application
==============================================================================================

setGeometry 
   called from BeginOfRunAction passing the world pointer over 
   in order to translate and serialize the geometry and copy it
   to the GPU  

addGenstep
   called from every step of modified Scintillation and Cerenkov processees
   in place of the optical photon generation loop, all the collected 
   gensteps are propagated together in a single GPU launch when propagate
   is called

propagateOpticalPhotons and getHits
   called from EndOfEventAction 

collectSecondaryPhotons
   invoked for example from L4Cerenkov::PostStepDoIt 


Notes
-------

* :doc:`notes/issues/G4OK`

**/

class G4OK_API G4Opticks   
{
        friend struct G4OKTest ; 
    private:
        static const plog::Severity LEVEL ;
        static const char* fEmbeddedCommandLine ; 
        static std::string EmbeddedCommandLine(const char* extra=NULL); 
        Opticks* InitOpticks(const char* keyspec);
    public:
        static G4Opticks* Get();
        static void Initialize(const char* gdmlpath, bool standardize_geant4_materials);
        static void Initialize(const G4VPhysicalVolume* world, bool standardize_geant4_materials);
    public:
        G4Opticks();
        virtual ~G4Opticks();
    public:
        std::string desc() const ;  
        const char* dbgdesc() const ;  
        std::string dbgdesc_() const ;
    public:
        // workflow methods
        int propagateOpticalPhotons(G4int eventID);
        NPY<float>* getHits() const ; 
        void reset(); 
        void setAlignIndex(int align_idx) const ; 

        static void Finalize();

    public:
        bool isLoadedFromCache() const ;
        unsigned getNumSensorVolumes() const ; 
        unsigned getSensorIdentityStandin(unsigned sensorIndex) const ;        // pre-cache and post-cache 
        const std::vector<G4PVPlacement*>& getSensorPlacements() const ;  // pre-cache live running only 
        unsigned getNumDistinctPlacementCopyNo() const  ; 

    public:
        void setSensorData(unsigned sensorIndex, float efficiency_1, float efficiency_2, int sensor_category, int sensor_identifier);
        void getSensorData(unsigned sensorIndex, float& efficiency_1, float& efficiency_2, int& category, int& identifier) const ;
        int  getSensorIdentifier(unsigned sensorIndex) const ;

        template <typename T> void setSensorDataMeta( const char* key, T value );
    public:
        void setSensorAngularEfficiency( const std::vector<int>& shape, const std::vector<float>& data, 
                                         int theta_steps, float theta_min, float theta_max, 
                                         int phi_steps=0,   float phi_min=0.f, float phi_max=0.f );

        template <typename T> void setSensorAngularEfficiencyMeta( const char* key, T value );
    public:
        NPY<float>*  getSensorDataArray() const ;
        NPY<float>*  getSensorAngularEfficiencyArray() const ;
    public:
        void saveSensorArrays(const char* dir) const ; 
    public:
        void setGeometry(const G4VPhysicalVolume* world); 
        void setGeometry(const char* gdmlpath);
        void setGeometry(const G4VPhysicalVolume* world, bool standardize_geant4_materials); 
        void loadGeometry(); 
    public:
        void setStandardizeGeant4Materials(bool standardize_geant4_materials);
    private:
        void setGeometry(const GGeo* ggeo); 
    private:
        GGeo* translateGeometry( const G4VPhysicalVolume* top );
        void initSensorData(unsigned sensor_num);

        void standardizeGeant4MaterialProperties();
        void createCollectors();
        void resetCollectors(); 
        void setupMaterialLookup();
    public:
        unsigned getNumPhotons() const ;
        unsigned getNumGensteps() const ;

        void collectGenstep_G4Cerenkov_1042(  
             const G4Track*  aTrack, 
             const G4Step*   aStep, 
             G4int       numPhotons,

             G4double    betaInverse,
             G4double    pmin,
             G4double    pmax,
             G4double    maxCos,

             G4double    maxSin2,
             G4double    meanNumberOfPhotons1,
             G4double    meanNumberOfPhotons2
            );


        /**
        2018/9/8 Geant4.1042 requires both velocities so:
             meanVelocity->preVelocity
             spare1->postVelocity 
             see notes/issues/ckm_cerenkov_generation_align_small_quantized_deviation_g4_g4.rst
        **/
        void collectCerenkovStep(
            G4int                id, 
            G4int                parentId,
            G4int                materialId,
            G4int                numPhotons,
    
            G4double             x0_x,  
            G4double             x0_y,  
            G4double             x0_z,  
            G4double             t0, 

            G4double             deltaPosition_x, 
            G4double             deltaPosition_y, 
            G4double             deltaPosition_z, 
            G4double             stepLength, 

            G4int                pdgCode, 
            G4double             pdgCharge, 
            G4double             weight, 
            G4double             preVelocity, 

            G4double             betaInverse,
            G4double             pmin,
            G4double             pmax,
            G4double             maxCos,

            G4double             maxSin2,
            G4double             meanNumberOfPhotons1,
            G4double             meanNumberOfPhotons2,
            G4double             postVelocity
        );



        void collectGenstep_G4Scintillation_1042(  
             const G4Track* aTrack, 
             const G4Step* aStep, 
             G4int    numPhotons, 
             G4int    ScintillationType,
             G4double ScintillationTime, 
             G4double ScintillationRiseTime
             );

        void collectGenstep_DsG4Scintillation_r3971(  
             const G4Track* aTrack, 
             const G4Step* aStep, 
             G4int    numPhotons, 
             G4int    scnt,          //  1:fast 2:slow
             G4double slowerRatio,
             G4double slowTimeConstant,
             G4double slowerTimeConstant,
             G4double ScintillationTime
            );


	   void collectScintillationStep(
            G4int id,
            G4int parentId,
            G4int materialId,
            G4int numPhotons,

            G4double x0_x,
            G4double x0_y,
            G4double x0_z,
            G4double t0,

            G4double deltaPosition_x,
            G4double deltaPosition_y,
            G4double deltaPosition_z,
            G4double stepLength,

            G4int pdgCode,
            G4double pdgCharge,
            G4double weight,
            G4double meanVelocity,

            G4int scntId,
            G4double slowerRatio,
            G4double slowTimeConstant,
            G4double slowerTimeConstant,

            G4double scintillationTime,
            G4double scintillationIntegrationMax,
            G4double spare1,
            G4double spare2
            );


    public:
        void collectSecondaryPhotons(const G4VParticleChange* pc);
    public:
        void collectHit(
            G4double             pos_x,  
            G4double             pos_y,  
            G4double             pos_z,  
            G4double             time ,

            G4double             dir_x,  
            G4double             dir_y,  
            G4double             dir_z,  
            G4double             weight ,

            G4double             pol_x,  
            G4double             pol_y,  
            G4double             pol_z,  
            G4double             wavelength ,

            G4int                flags_x, 
            G4int                flags_y, 
            G4int                flags_z, 
            G4int                flags_w
         ); 

        void getHit(
            unsigned i,
            G4ThreeVector* position,  
            G4double* time, 
            G4ThreeVector* direction, 
            G4double* weight,
            G4ThreeVector* polarization,  
            G4double* wavelength, 
            G4int* flags_x,
            G4int* flags_y,
            G4int* flags_z,
            G4int* flags_w,
            G4bool* is_cerenkov, 
            G4bool* is_reemission,
            G4int*  sensor_index,
            G4int*  sensor_identifier 
       ) const  ; 
            
     private:
        bool                       m_standardize_geant4_materials ; 
        const G4VPhysicalVolume*   m_world ; 
        const GGeo*                m_ggeo ; 
        const GBndLib*             m_blib ; 
        Opticks*                   m_ok ;
        CTraverser*                m_traverser ; 
        CMaterialTable*            m_mtab ; 
        CGenstepCollector*         m_genstep_collector ; 
        CPrimaryCollector*         m_primary_collector ; 
        NLookup*                   m_lookup ; 
        OpMgr*                     m_opmgr;
    private:
        // transient pointers borrowed from the collectors 
        NPY<float>*                m_gensteps ; 
        NPY<float>*                m_genphotons ; 
        NPY<float>*                m_hits_ ; 
        NPho*                      m_hits ; 
        unsigned                   m_num_hits ; 
    private:
        // minimal instrumentation from the G4 side of things 
        CPhotonCollector*          m_g4hit_collector ; 
        C4PhotonCollector*         m_g4photon_collector ; 
        unsigned                   m_genstep_idx ; 
    private:
        OpticksEvent*              m_g4evt ; 
        NPY<float>*                m_g4hit ; 
        bool                       m_gpu_propagate ;  
    private:
        std::vector<G4PVPlacement*> m_sensor_placements ;
        unsigned                    m_sensor_num ; 
        NPY<float>*                 m_sensor_data ; 
        NPY<float>*                 m_sensor_angular_efficiency ; 
    private:
        static G4Opticks*          fInstance;

};



