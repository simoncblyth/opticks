#include "OpSeeder.hh"

// npy-
#include "Timer.hpp"
#include "NLog.hpp"
#include "NumpyEvt.hpp"
#include "NPY.hpp"

// cudawrap-
#include "CResource.hh"
#include "CBufSpec.hh"

// thrustrap-
#include "TBuf.hh"
#include "TBufPair.hh"


#define TIMER(s) \
    { \
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
    }


void OpSeeder::init()
{
}

void OpSeeder::seedPhotonsFromGenstepsViaOpenGL()
{
    LOG(info)<<"OpSeeder::seedPhotonsFromGenstepsViaOpenGL" ;

    NPY<float>* gensteps =  m_evt->getGenstepData() ;
    NPY<float>* photons  =  m_evt->getPhotonData() ;    // NB has no allocation and "uploaded" with glBufferData NULL

    unsigned int num_values = gensteps->getNumValues(0) ; 

    int gensteps_id = gensteps->getBufferId() ;
    int photons_id = photons->getBufferId() ; 

    assert(gensteps_id > -1);
    assert(photons_id > -1);

    CResource rgs( gensteps_id , CResource::R );
    CResource rph( photons_id, CResource::RW );

    CBufSpec rgs_ = rgs.mapGLToCUDA<unsigned int>() ;
    CBufSpec rph_ = rph.mapGLToCUDA<unsigned int>() ;

    TBuf tgs("tgs", rgs_ );
    TBuf tph("tph", rph_ );
    
    //tgs.dump<unsigned int>("App::seedPhotonsFromGensteps tgs", 6*4, 3, nv0 ); // stride, begin, end 

    unsigned int num_photons = tgs.reduce<unsigned int>(6*4, 3, num_values );  // adding photon counts for each genstep 

    unsigned int x_num_photons = m_evt->getNumPhotons() ;

    if(num_photons != x_num_photons)
          LOG(fatal)
          << "OpSeeder::seedPhotonsFromGensteps"
          << " num_photons " << num_photons 
          << " x_num_photons " << x_num_photons 
          ;

    assert(num_photons == x_num_photons && "FATAL : mismatch between CPU and GPU photon counts from the gensteps") ;   

    CBufSlice src = tgs.slice(6*4,3,num_values) ;
    CBufSlice dst = tph.slice(4*4,0,num_photons*4*4) ;

    TBufPair<unsigned int> tgp(src, dst);
    tgp.seedDestination();

    rgs.unmapGLToCUDA(); 
    rph.unmapGLToCUDA(); 

    TIMER("seedPhotonsFromGenstepsViaOpenGL"); 
}


void OpSeeder::seedPhotonsFromGensteps()
{
    LOG(info)<<"OpSeeder::seedPhotonsFromGensteps" ;

}
