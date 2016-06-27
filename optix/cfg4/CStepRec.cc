// okc-
#include "OpticksEvent.hh"

// npy-
#include "NPY.hpp"
#include "uif.h"

// cg4-
#include "CStep.hh"
#include "CStepRec.hh"
#include "Format.hh"

// g4-
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "PLOG.hh"


CStepRec::CStepRec( OpticksEvent* evt )
   :
   m_evt(evt),
   m_nopstep(NULL),
   m_store_count(0)
{
    init();
}

unsigned int CStepRec::getStoreCount()
{
   return m_store_count ; 
}



void CStepRec::init()
{
    m_nopstep = m_evt->getNopstepData();
}

void CStepRec::collectStep(const G4Step* step, unsigned int step_id)
{
    m_steps.push_back(new CStep(step, step_id ));
}

void CStepRec::storeStepsCollected(unsigned int event_id, unsigned int track_id, int particle_id)
{
    m_store_count += 1 ; 
    unsigned int nsteps = m_steps.size();

    LOG(info) << "CStepRec::storeStepsCollected" 
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
         unsigned int step_id = cstep->getStepId();
         //assert(step_id == i);

         storePoint(event_id, track_id, particle_id, i, step->GetPreStepPoint() ) ;

         if( i == nsteps - 1) 
             storePoint(event_id, track_id, particle_id, i+1, step->GetPostStepPoint() );
    }

    m_steps.clear();
}


void CStepRec::storePoint(unsigned int event_id, unsigned int track_id, int particle_id, unsigned int point_id, const G4StepPoint* point)
{
    const G4ThreeVector& pos = point->GetPosition();
    G4double time = point->GetGlobalTime();

    const G4ThreeVector& dir = point->GetMomentumDirection();
    G4double weight = 1.0 ; 

    const G4ThreeVector& pol = point->GetPolarization();
    G4double energy = point->GetKineticEnergy();

    unsigned int nvals = 16 ; 
    float* vals = new float[nvals] ;

    vals[0] =  pos.x()/mm ;  
    vals[1] =  pos.y()/mm ;  
    vals[2] =  pos.z()/mm ;  
    vals[3] =  time/ns    ;  

    vals[4] =  dir.x() ;
    vals[5] =  dir.y() ;
    vals[6] =  dir.z() ;
    vals[7] =  weight ;

    vals[8]  =  pol.x() ;
    vals[9]  =  pol.y() ;
    vals[10] =  pol.z() ;
    vals[11] =  energy/keV ;

    uif_t uif[4] ; 

    uif[0].u = event_id ;
    uif[1].u = track_id ; 
    uif[2].i = particle_id ; 
    uif[3].u = point_id ;

    vals[12]  = uif[0].f ;
    vals[13]  = uif[1].f ;
    vals[14] =  uif[2].f ;
    vals[15] =  uif[3].f ;

    m_nopstep->add(vals, nvals);
    // dynamically adding is efficient, but cannot handle on GPU

    delete vals ; 

}




