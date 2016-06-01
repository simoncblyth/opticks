#include "OpSeeder.hh"

// optickscore-
#include "OpticksEvent.hh"

// npy-
#include "Timer.hpp"
#include "NLog.hpp"
#include "NPY.hpp"

// cudawrap-
#include "CResource.hh"
#include "CBufSpec.hh"

// thrustrap-
#include "TBuf.hh"
#include "TBufPair.hh"

// optixrap-
#include "OContext.hh"
#include "OPropagator.hh"
#include "OBuf.hh"


#define TIMER(s) \
    { \
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
    }



void OpSeeder::seedPhotonsFromGensteps()
{
    LOG(info)<<"OpSeeder::seedPhotonsFromGensteps" ;
    if( m_ocontext->isInterop() )
    {    
        seedPhotonsFromGenstepsViaOpenGL();
    }    
    else if ( m_ocontext->isCompute() )
    {    
        seedPhotonsFromGenstepsViaOptiX();
    }    
}


void OpSeeder::seedPhotonsFromGenstepsViaOpenGL()
{
    LOG(info)<<"OpSeeder::seedPhotonsFromGenstepsViaOpenGL" ;

    NPY<float>* gensteps =  m_evt->getGenstepData() ;
    NPY<float>* photons  =  m_evt->getPhotonData() ;    // NB has no allocation and "uploaded" with glBufferData NULL

    int gensteps_id = gensteps->getBufferId() ;
    int photons_id = photons->getBufferId() ; 

    assert(gensteps_id > -1);
    assert(photons_id > -1);

    CResource r_gs( gensteps_id , CResource::R );
    CResource r_ox( photons_id, CResource::RW );

    CBufSpec s_gs = r_gs.mapGLToCUDA<unsigned int>() ;
    CBufSpec s_ox = r_ox.mapGLToCUDA<unsigned int>() ;

    seedPhotonsFromGenstepsImp(s_gs, s_ox);

    r_gs.unmapGLToCUDA(); 
    r_ox.unmapGLToCUDA(); 

    TIMER("seedPhotonsFromGenstepsViaOpenGL"); 
}

void OpSeeder::seedPhotonsFromGenstepsViaOptiX()
{
    assert(m_propagator);
 
    OBuf* genstep = m_propagator->getGenstepBuf() ;
    OBuf* photon = m_propagator->getPhotonBuf() ;

    CBufSpec s_gs = genstep->bufspec();
    CBufSpec s_ox = photon->bufspec();

    seedPhotonsFromGenstepsImp(s_gs, s_ox);

    TIMER("seedPhotonsFromGenstepsViaOptiX"); 
}


void OpSeeder::seedPhotonsFromGenstepsImp(const CBufSpec& s_gs, const CBufSpec& s_ox)
{
    TBuf tgs("tgs", s_gs );
    TBuf tox("tox", s_ox );
    
    //tgs.dump<unsigned int>("App::seedPhotonsFromGenstepsImp tgs", 6*4, 3, nv0 ); // stride, begin, end 

    NPY<float>* gensteps =  m_evt->getGenstepData() ;

    unsigned int num_values = gensteps->getNumValues(0) ; 

    unsigned int num_photons = tgs.reduce<unsigned int>(6*4, 3, num_values );  // adding photon counts for each genstep 

    unsigned int x_num_photons = m_evt->getNumPhotons() ;

    if(num_photons != x_num_photons)
          LOG(fatal)
          << "OpSeeder::seedPhotonsFromGenstepsImp"
          << " num_photons " << num_photons 
          << " x_num_photons " << x_num_photons 
          ;

    assert(num_photons == x_num_photons && "FATAL : mismatch between CPU and GPU photon counts from the gensteps") ;   

    CBufSlice src = tgs.slice(6*4,3,num_values) ;
    CBufSlice dst = tox.slice(4*4,0,num_photons*4*4) ;

    TBufPair<unsigned int> tgp(src, dst);
    tgp.seedDestination();
}


