#include <cstddef>

#include "OpticksBufferControl.hh"  // okc-
#include "OpticksSwitches.h"  
#include "Opticks.hh"  
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





OpSeeder::OpSeeder(OpticksHub* hub, OEvent* oevt)  
   :
     m_hub(hub),
     m_ok(hub->getOpticks()),
     m_oevt(oevt),
     m_ocontext(oevt->getOContext())
{
}


void OpSeeder::seedPhotonsFromGensteps()
{
    LOG(info)<<"OpSeeder::seedPhotonsFromGensteps" ;
    if( m_ocontext->isInterop() )
    {    
#ifdef WITH_SEED_BUFFER
        seedComputeSeedsFromInteropGensteps();
#else
        seedPhotonsFromGenstepsViaOpenGL();
#endif
    }    
    else if ( m_ocontext->isCompute() )
    {    
        seedPhotonsFromGenstepsViaOptiX();
    }    
    if(m_hub->hasOpt("onlyseed")) exit(EXIT_SUCCESS);
}

void OpSeeder::seedComputeSeedsFromInteropGensteps()
{
#ifdef WITH_SEED_BUFFER
    LOG(info)<<"OpSeeder::seedComputeSeedsFromInteropGensteps : WITH_SEED_BUFFER " ;

    OpticksEvent* evt = m_hub->getEvent();
    assert(evt); 

    NPY<unsigned>* seedData =  evt->getSeedData() ;
    assert(seedData->isComputeBuffer());

    OBuf* seed = m_oevt->getSeedBuf() ;
    CBufSpec s_se = seed->bufspec();


    NPY<float>* gensteps =  evt->getGenstepData() ;
    int gensteps_id = gensteps->getBufferId() ;
    assert(gensteps_id > -1 );

    CResource r_gs( gensteps_id , CResource::R );
    CBufSpec s_gs = r_gs.mapGLToCUDA<unsigned int>() ;
    s_gs.size = s_gs.size/4 ; 

    seedPhotonsFromGenstepsImp(s_gs, s_se);

    r_gs.unmapGLToCUDA(); 

#else
    assert(0 && "OpSeeder::seedComputeSeedsFromInteropGensteps is applicable only WITH_SEED_BUFFER "); 
#endif
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
    OK_PROFILE("_OpSeeder::seedPhotonsFromGenstepsViaOptiX");

    OBuf* genstep = m_oevt->getGenstepBuf() ;
    CBufSpec s_gs = genstep->bufspec();

#ifdef WITH_SEED_BUFFER
    LOG(info) << "OpSeeder::seedPhotonsFromGenstepsViaOptiX : SEEDING TO SEED BUF  " ; 
    OBuf* seed = m_oevt->getSeedBuf() ;
    CBufSpec s_se = seed->bufspec();
    seedPhotonsFromGenstepsImp(s_gs, s_se);
    //s_gs.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_gs");
    //s_se.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_se");
#else
    LOG(info) << "OpSeeder::seedPhotonsFromGenstepsViaOptiX : seeding to photon buf  " ; 
    OBuf* photon = m_oevt->getPhotonBuf() ;
    CBufSpec s_ox = photon->bufspec();
    seedPhotonsFromGenstepsImp(s_gs, s_ox);
#endif

    //genstep->Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (OBuf)genstep");
    //s_gs.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_gs");

    //photon->Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (OBuf)photon ");
    //s_ox.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_ox");



    TIMER("seedPhotonsFromGenstepsViaOptiX"); 
    OK_PROFILE("OpSeeder::seedPhotonsFromGenstepsViaOptiX");

}




unsigned OpSeeder::getNumPhotonsCheck(const TBuf& tgs)
{
    OpticksEvent* evt = m_hub->getEvent();

    assert(evt); 

    NPY<float>* gensteps =  evt->getGenstepData() ;

    unsigned int num_genstep_values = gensteps->getNumValues(0) ; 

    unsigned int num_photons = tgs.reduce<unsigned int>(6*4, 3, num_genstep_values );  // adding photon counts for each genstep 
 
    unsigned int x_num_photons = evt->getNumPhotons() ;


    if(num_photons != x_num_photons)
          LOG(fatal)
          << "OpSeeder::getNumPhotonsCheck"
          << " num_photons " << num_photons 
          << " x_num_photons " << x_num_photons 
          ;

    assert(num_photons == x_num_photons && "FATAL : mismatch between CPU and GPU photon counts from the gensteps") ;   

    return num_photons ; 
}




void OpSeeder::seedPhotonsFromGenstepsImp(const CBufSpec& s_gs, const CBufSpec& s_ox)
{
    //s_gs.Summary("OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_gs");
    //s_ox.Summary("OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_ox");

    TBuf tgs("tgs", s_gs );
    TBuf tox("tox", s_ox );
    

    OpticksEvent* evt = m_hub->getEvent();
    assert(evt); 

    NPY<float>* gensteps =  evt->getGenstepData() ;

    unsigned int num_genstep_values = gensteps->getNumValues(0) ; 

    //tgs.dump<unsigned int>("OpSeeder::seedPhotonsFromGenstepsImp tgs.dump", 6*4, 3, num_genstep_values ); // stride, begin, end 


    unsigned int num_photons = getNumPhotonsCheck(tgs);

    OpticksBufferControl* ph_ctrl = evt->getPhotonCtrl();

    if(ph_ctrl->isSet("VERBOSE_MODE"))
    LOG(info) << "OpSeeder::seedPhotonsFromGenstepsImp photons(VERBOSE_MODE) "
               << " num_photons " << num_photons 
               << " gensteps " << gensteps->getShapeString() 
               << " num_genstep_values " << num_genstep_values
               ;

    // src slice is plucking photon counts from each genstep
    // dst slice points at the first value of each item in photon buffer
    // buffer size and num_bytes comes directly from CBufSpec
    CBufSlice src = tgs.slice(6*4,3,num_genstep_values) ;  // stride, begin, end 

#ifdef WITH_SEED_BUFFER
    tox.zero();   // huh seeding of SEED buffer requires zeroing ahead ?? otherwise get one 0 with the rest 4294967295 ie overrun -1 
    CBufSlice dst = tox.slice(1*1,0,num_photons*1*1) ;
#else
    CBufSlice dst = tox.slice(4*4,0,num_photons*4*4) ;
#endif

    bool verbose = false ; 
    TBufPair<unsigned int> tgp(src, dst, verbose);
    tgp.seedDestination();

#ifdef WITH_SEED_BUFFER
    //tox.dump<unsigned int>("OpSeeder::seedPhotonsFromGenstepsImp tox.dump", 1*1, 0, num_photons ); // stride, begin, end 
#endif

}

/*
In [5]: np.uint32((1 << 32) - 1 )
Out[5]: 4294967295

In [6]: np.uint32(-1)
Out[6]: 4294967295


*/






