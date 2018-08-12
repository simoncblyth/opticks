#pragma once

#include <string>
#include "G4Types.hh"

//class OpticksHub ; 
class NLookup ; 
template <typename T> class NPY ;

/**
CCollector
==================== 

Canonical instances:

1. CG4.m_collector instanciated at postinitialize 



Gensteps have an item shape of 6*4 (ie 6 quads) the 
first 3 quads are common between scintillation and
Cerenkov but the last 3 differ.

Furthermore the precise details of which quanties are included 
in the last three quads depends on the inner photon loop 
implementation details of the scintillation and Cerenkov processes.
They need to be setup whilst developing a corresponding GPU/OptiX 
implementation of the inner photon loop and doing 
equivalence comparisons.

Each implementation will need slightly different genstep and OptiX port.

Effectively the genstep can be regarded as the "stack" just 
prior to the photon loop.
 
**/

#include "CFG4_API_EXPORT.hh"

class CFG4_API CCollector 
{
   public:
         static CCollector* Instance();
   public:
         CCollector(const NLookup* lookup);  
   public:
         NPY<float>*  getGensteps();
         NPY<float>*  getPrimary();
   public:
         std::string description();
         void Summary(const char* msg="CCollector::Summary");
         int translate(int acode);
   private:
         void setGensteps(NPY<float>* gs);
         void consistencyCheck();
   public:
         void collectScintillationStep(
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
            G4double             meanVelocity, 

            G4int                scntId,
            G4double             slowerRatio,
            G4double             slowTimeConstant,
            G4double             slowerTimeConstant,

            G4double             scintillationTime,
            G4double             scintillationIntegrationMax,
            G4double             spare1=0,
            G4double             spare2=0
        );
   public:
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
            G4double             meanVelocity, 

            G4double             betaInverse,
            G4double             pmin,
            G4double             pmax,
            G4double             maxCos,

            G4double             maxSin2,
            G4double             meanNumberOfPhotons1,
            G4double             meanNumberOfPhotons2,
            G4double             spare2=0
        );
   public:
          void collectPrimary(
               G4double  x0,
               G4double  y0,
               G4double  z0,
               G4double  t0,

               G4double  dir_x,
               G4double  dir_y,
               G4double  dir_z,
               G4double  weight,

               G4double  pol_x,
               G4double  pol_y,
               G4double  pol_z,
               G4double  wavelength,

               unsigned flags_x,
               unsigned flags_y,
               unsigned flags_z,
               unsigned flags_w
          );

   public:
         void collectMachineryStep(unsigned code);
   private:
         static CCollector* INSTANCE ;      
   private:
         const NLookup*     m_lookup ; 

         NPY<float>*  m_genstep ;
         unsigned     m_genstep_itemsize ; 
         float*       m_genstep_values ;  

         unsigned     m_scintillation_count ; 
         unsigned     m_cerenkov_count ; 
         unsigned     m_machinery_count ; 

         NPY<float>*  m_primary ;
         unsigned     m_primary_itemsize ; 
         float*       m_primary_values ;  
         unsigned     m_primary_count ;  

};
