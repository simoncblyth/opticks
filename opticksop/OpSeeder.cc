#include <cstddef>

#include "OpticksBufferControl.hh"  // okc-
#include "OpticksEvent.hh"  
#include "OpticksHub.hh"    // okg-

#include "Timer.hpp"   // npy-
#include "NPY.hpp"

// cudawrap-
#include "CResource.hh"
#include "CBufSpec.hh"

// thrustrap-
#include "TBuf.hh"
#include "TBufPair.hh"

// optixrap-
#include "OContext.hh"
#include "OEvent.hh"
#include "OBuf.hh"

#include "OpSeeder.hh"

#include "PLOG.hh"

#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }



//#define WITH_SEED_BUF 1


OpSeeder::OpSeeder(OpticksHub* hub, OEvent* oevt)  
   :
     m_hub(hub),
     m_oevt(oevt),
     m_ocontext(oevt->getOContext())
{
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
    if(m_hub->hasOpt("onlyseed")) exit(EXIT_SUCCESS);
}


void OpSeeder::seedPhotonsFromGenstepsViaOpenGL()
{
    LOG(info)<<"OpSeeder::seedPhotonsFromGenstepsViaOpenGL" ;

    OpticksEvent* evt = m_hub->getEvent();
    assert(evt); 
    NPY<float>* gensteps =  evt->getGenstepData() ;
    NPY<float>* photons  =  evt->getPhotonData() ;    // NB has no allocation and "uploaded" with glBufferData NULL

    int gensteps_id = gensteps->getBufferId() ;
    int photons_id = photons->getBufferId() ; 

    assert(gensteps_id > -1);
    assert(photons_id > -1);

    CResource r_gs( gensteps_id , CResource::R );
    CResource r_ox( photons_id, CResource::RW );

    CBufSpec s_gs = r_gs.mapGLToCUDA<unsigned int>() ;
    CBufSpec s_ox = r_ox.mapGLToCUDA<unsigned int>() ;

    // for matching with compute mode which gets the 
    // 4 from the multiplity of float4 optix buffer in OBuf
    s_gs.size = s_gs.size/4 ; 
    s_ox.size = s_ox.size/4 ; 

    seedPhotonsFromGenstepsImp(s_gs, s_ox);

    r_gs.unmapGLToCUDA(); 
    r_ox.unmapGLToCUDA(); 

    TIMER("seedPhotonsFromGenstepsViaOpenGL"); 
}

void OpSeeder::seedPhotonsFromGenstepsViaOptiX()
{
    OBuf* genstep = m_oevt->getGenstepBuf() ;
    CBufSpec s_gs = genstep->bufspec();

#ifdef WITH_SEED_BUF
    LOG(warning) << "OpSeeder::seedPhotonsFromGenstepsViaOptiX : SEEDING TO SEED BUF  " ; 
    OBuf* seed = m_oevt->getSeedBuf() ;
    CBufSpec s_ox = seed->bufspec();
#else
    LOG(info) << "OpSeeder::seedPhotonsFromGenstepsViaOptiX : seeding to photon buf  " ; 
    OBuf* photon = m_oevt->getPhotonBuf() ;
    CBufSpec s_ox = photon->bufspec();
#endif

    //genstep->Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (OBuf)genstep");
    //s_gs.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_gs");

    //photon->Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (OBuf)photon ");
    //s_ox.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_ox");


    seedPhotonsFromGenstepsImp(s_gs, s_ox);

    TIMER("seedPhotonsFromGenstepsViaOptiX"); 

}


void OpSeeder::seedPhotonsFromGenstepsImp(const CBufSpec& s_gs, const CBufSpec& s_ox)
{
    //s_gs.Summary("OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_gs");
    //s_ox.Summary("OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_ox");

    TBuf tgs("tgs", s_gs );
    TBuf tox("tox", s_ox );
    
    //tgs.dump<unsigned int>("App::seedPhotonsFromGenstepsImp tgs", 6*4, 3, nv0 ); // stride, begin, end 

    OpticksEvent* evt = m_hub->getEvent();
    assert(evt); 

    NPY<float>* gensteps =  evt->getGenstepData() ;
    NPY<float>* photons  =  evt->getPhotonData() ;

    unsigned int num_genstep_values = gensteps->getNumValues(0) ; 

    OpticksBufferControl ph_ctrl(photons->getBufferControlPtr());
    bool ph_verbose = ph_ctrl("VERBOSE_MODE") ;

    if(ph_verbose)
    LOG(info) << "OpSeeder::seedPhotonsFromGenstepsImp photons(VERBOSE_MODE) "
               << " photons " << photons->getShapeString() 
               << " gensteps " << gensteps->getShapeString() 
               << " num_genstep_values " << num_genstep_values
               ;

    unsigned int num_photons = tgs.reduce<unsigned int>(6*4, 3, num_genstep_values );  // adding photon counts for each genstep 

    unsigned int x_num_photons = evt->getNumPhotons() ;

    if(num_photons != x_num_photons)
          LOG(fatal)
          << "OpSeeder::seedPhotonsFromGenstepsImp"
          << " num_photons " << num_photons 
          << " x_num_photons " << x_num_photons 
          ;

    assert(num_photons == x_num_photons && "FATAL : mismatch between CPU and GPU photon counts from the gensteps") ;   

    // src slice is plucking photon counts from each genstep
    // dst slice points at the first value of each item in photon buffer
    // buffer size and num_bytes comes directly from CBufSpec
    CBufSlice src = tgs.slice(6*4,3,num_genstep_values) ;  // stride, begin, end 

#ifdef WITH_SEED_BUF
    CBufSlice dst = tox.slice(1*1,0,num_photons*1*1) ;
#else
    CBufSlice dst = tox.slice(4*4,0,num_photons*4*4) ;
#endif

    TBufPair<unsigned int> tgp(src, dst);
    tgp.seedDestination();


}


