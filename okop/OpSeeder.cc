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

#include <cstddef>
#include <algorithm>

#include "OpticksBufferControl.hh"  // okc-
#include "OpticksSwitches.h"  
#include "Opticks.hh"  
#include "OpticksEvent.hh"  

#include "BTimeKeeper.hh"   // npy-
#include "NPY.hpp"

// cudarap-
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

const plog::Severity OpSeeder::LEVEL = PLOG::EnvLevel("OpSeeder", "DEBUG") ; 

OpSeeder::OpSeeder(Opticks* ok, OEvent* oevt)  
    :
    m_ok(ok),
    m_dbg(m_ok->hasOpt("dbgseed")),
    m_oevt(oevt),
    m_ocontext(oevt->getOContext())
{
}


void OpSeeder::seedPhotonsFromGensteps()
{
    LOG(debug)<<"OpSeeder::seedPhotonsFromGensteps" ;
    if( m_ocontext->isCompute() )
    {    
        seedPhotonsFromGenstepsViaOptiX();
    }    
    else if( m_ocontext->isInterop() )
    {    
#ifdef WITH_SEED_BUFFER
        seedComputeSeedsFromInteropGensteps();
#else
        seedPhotonsFromGenstepsViaOpenGL();
#endif
    }    

   // if(m_ok->hasOpt("onlyseed")) exit(EXIT_SUCCESS);
}

void OpSeeder::seedComputeSeedsFromInteropGensteps()
{
#ifdef WITH_SEED_BUFFER
    LOG(info)<<"OpSeeder::seedComputeSeedsFromInteropGensteps : WITH_SEED_BUFFER " ;

    OpticksEvent* evt = m_ok->getEvent();
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

    OpticksEvent* evt = m_ok->getEvent();
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

}


/**
OpSeeder::seedPhotonsFromGenstepsViaOptiX
-------------------------------------------

Access two GPU buffers via OEvent m_oevt and OBuf::

1. genstep buffer
2. seed buffer OR photon buffer  
3. apply seedPhotonsFromGenstepsImp

Seeding to the photon buf has the disadvantage that 
you need to write to it as well as read, whereas when
seeding to the seed buffer the photon buffer becomes 
read only from CPU side.

Seeding to SEED BUF is the current default.

**/

void OpSeeder::seedPhotonsFromGenstepsViaOptiX()
{
    OK_PROFILE("_OpSeeder::seedPhotonsFromGenstepsViaOptiX");

    OBuf* genstep = m_oevt->getGenstepBuf() ;
    CBufSpec s_gs = genstep->bufspec();

#ifdef WITH_SEED_BUFFER
    LOG(LEVEL) << "SEEDING TO SEED BUF  " ; 
    OBuf* seed = m_oevt->getSeedBuf() ;
    CBufSpec s_se = seed->bufspec();   //  optix::Buffer::getDevicePointer happens here  ( CBufSpec just holder for devPtr, size, numBytes )
    seedPhotonsFromGenstepsImp(s_gs, s_se);
    //s_gs.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_gs");
    //s_se.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_se");
#else
    LOG(info) << "seeding to photon buf  " ; 
    OBuf* photon = m_oevt->getPhotonBuf() ;
    CBufSpec s_ox = photon->bufspec();
    seedPhotonsFromGenstepsImp(s_gs, s_ox);
#endif

    //genstep->Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (OBuf)genstep");
    //s_gs.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_gs");

    //photon->Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (OBuf)photon ");
    //s_ox.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_ox");

    OK_PROFILE("OpSeeder::seedPhotonsFromGenstepsViaOptiX");

}




unsigned OpSeeder::getNumPhotonsCheck(const TBuf& tgs)
{
    OpticksEvent* evt = m_ok->getEvent();
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


/**
OpSeeder::seedPhotonsFromGenstepsImp
--------------------------------------

1. create TBuf (Thrust buffer accessors) for the two buffers
2. access CPU side gensteps from OpticksEvent
3. check the photon counts from the GPU side gensteps match those from CPU side
   (this implies that the event gensteps must have been uploaded to GPU already)
4. create src(photon counts per genstep) and dst(genstep indices) buffer slices
   with appropriate strides and offsets 
5. use TBufPair::seedDestination which distributes genstep indices to every photon

**/

void OpSeeder::seedPhotonsFromGenstepsImp(const CBufSpec& s_gs, const CBufSpec& s_ox)
{
    if(m_dbg)
    { 
        s_gs.Summary("OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_gs --dbgseed");
        s_ox.Summary("OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_ox --dbgseed");
    }

    TBuf tgs("tgs", s_gs, " ");   // ctor just copies
    TBuf tox("tox", s_ox, " ");
    

    OpticksEvent* evt = m_ok->getEvent();
    assert(evt); 

    NPY<float>* gensteps =  evt->getGenstepData() ;

    unsigned num_genstep_values = gensteps->getNumValues(0) ; 

    if(m_dbg)
    {
       LOG(info) << "OpSeeder::seedPhotonsFromGenstepsImp"
                 << " gensteps " << gensteps->getShapeString()
                 << " num_genstep_values " << num_genstep_values
                 ;
       tgs.dump<unsigned>("OpSeeder::seedPhotonsFromGenstepsImp tgs.dump --dbgseed", 6*4, 3, num_genstep_values ); // stride, begin, end 
       // the last element of the first quad in the 6 quads of genstep is the number of photons 
    }


    unsigned num_photons = getNumPhotonsCheck(tgs);  // compare GPU reduction result on the gensteps with CPU num_photons 

    OpticksBufferControl* ph_ctrl = evt->getPhotonCtrl();

    if(ph_ctrl->isSet("VERBOSE_MODE") || m_dbg)
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


    bool verbose = m_dbg ; 
    TBufPair<unsigned> tgp(src, dst, verbose);
    tgp.seedDestination();

#ifdef WITH_SEED_BUFFER
    if(m_dbg)
    {
        tox.dump<unsigned>("OpSeeder::seedPhotonsFromGenstepsImp tox.dump --dbgseed", 1*1, 0, std::min(num_photons,10000u) ); // stride, begin, end 
    }
#endif

}



