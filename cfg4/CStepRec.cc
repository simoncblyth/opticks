#include "CFG4_BODY.hh"
// okc-
#include "OpticksEvent.hh"

// okg-
#include "OpticksHub.hh"

// npy-
#include "NPY.hpp"
#include "uif.h"

// cfg4-
#include "CStep.hh"
#include "CStepRec.hh"
#include "Format.hh"

// g4-
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "PLOG.hh"


CStepRec::CStepRec( Opticks* ok, bool dynamic)
   :
   m_ok(ok),
   m_dynamic(dynamic),
   m_store_count(0),
   m_num_vals(16), 
   m_vals(new float[m_num_vals]),
   m_nopstep(NULL)
{
}

unsigned int CStepRec::getStoreCount()
{
   return m_store_count ; 
}

void CStepRec::collectStep(const G4Step* step, unsigned int step_id)
{
    m_steps.push_back(new CStep(step, step_id ));  // CStep::CStep copies the step 
}

void CStepRec::storeStepsCollected(unsigned int event_id, unsigned int track_id, int particle_id)
{
    m_store_count += 1 ; 
    unsigned int nsteps = m_steps.size();

    LOG(debug) << "CStepRec::storeStepsCollected" 
              << " store_count " << m_store_count 
              << " event_id " << event_id
              << " track_id " << track_id
              << " particle_id " << particle_id
              << " nsteps " << nsteps 
              << "\n"
              << Format( m_steps , "steps", false )
              ;

    for(unsigned int i=0 ; i < nsteps ; i++)
    {
         const CStep* cstep = m_steps[i] ;
         const G4Step* step = cstep->getStep();
         //unsigned int step_id = cstep->getStepId();
         //assert(step_id == i);

         storePoint(event_id, track_id, particle_id, i, step->GetPreStepPoint() ) ;

         if( i == nsteps - 1) 
             storePoint(event_id, track_id, particle_id, i+1, step->GetPostStepPoint() );
    }

    m_steps.clear();
}

void CStepRec::initEvent(NPY<float>* nopstep)
{
    setNopstep(nopstep);
}

void CStepRec::setNopstep(NPY<float>* nopstep)
{
    m_nopstep = nopstep ; 
}

void CStepRec::storePoint(unsigned int event_id, unsigned int track_id, int particle_id, unsigned int point_id, const G4StepPoint* point)
{
    // nopstep updated when new G4 evt is created, so no action is required to handle change of event

    const G4ThreeVector& pos = point->GetPosition();
    G4double time = point->GetGlobalTime();

    const G4ThreeVector& dir = point->GetMomentumDirection();
    G4double weight = 1.0 ; 

    const G4ThreeVector& pol = point->GetPolarization();
    G4double energy = point->GetKineticEnergy();

    m_vals[0] =  pos.x()/mm ;  
    m_vals[1] =  pos.y()/mm ;  
    m_vals[2] =  pos.z()/mm ;  
    m_vals[3] =  time/ns    ;  

    m_vals[4] =  dir.x() ;
    m_vals[5] =  dir.y() ;
    m_vals[6] =  dir.z() ;
    m_vals[7] =  weight ;

    m_vals[8]  =  pol.x() ;
    m_vals[9]  =  pol.y() ;
    m_vals[10] =  pol.z() ;
    m_vals[11] =  energy/keV ;

    uif_t uif[4] ; 

    uif[0].u = event_id ;
    uif[1].u = track_id ; 
    uif[2].i = particle_id ; 
    uif[3].u = point_id ;

    m_vals[12]  = uif[0].f ;
    m_vals[13]  = uif[1].f ;
    m_vals[14] =  uif[2].f ;
    m_vals[15] =  uif[3].f ;

    m_nopstep->add(m_vals, m_num_vals);
}


