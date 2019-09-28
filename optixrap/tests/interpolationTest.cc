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

#include "SSys.hh"
#include "BFile.hh"
#include "NPY.hpp"

#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"

#include "OBndLib.hh"
#include "OLaunchTest.hh"
#include "OConfig.hh"
#include "OContext.hh"
#include "Opticks.hh"
#include "OpticksHub.hh"

#include "OScene.hh"

#include "OPTICKS_LOG.hh"


/**
interpolationTest
===================

NB the python scripts compare results from this interpolationtest with those from 
cfg4/tests/CInterpolationTest.cc 

Which mixes up dependency order, as CInterpolationTest is run after interpolationTest.

TODO:

* rearrange the python interpolationTest scripts to eliminate duplication and
  reposition to avoid skipping or fails ion first run due to funny ordering 


**/

class interpolationTest 
{
    public:
        interpolationTest(Opticks* ok, OContext* ocontext, OBndLib* obnd, const char* base);
        void launch(optix::Context& context);
        int ana();
        std::string desc() const ; 
    private:
        void init();    
    private:
        Opticks*    m_ok ;
        bool        m_interpol ; 
        const char* m_progname ; 
        const char* m_name ; 
        const char* m_script ; 
        OContext*   m_ocontext;
        OBndLib*    m_obnd ;
        const char* m_base ;  
        unsigned    m_nb ; 
        unsigned    m_nx ; 
        unsigned    m_ny ; 
        NPY<float>* m_out ; 
};



interpolationTest::interpolationTest(Opticks* ok, OContext* ocontext, OBndLib* obnd, const char* base)
    :
    m_ok(ok),
    m_interpol(!ok->hasOpt("nointerpol")),
    m_progname( m_interpol ? "interpolationTest" : "identityTest" ),
    m_name(NULL),
    m_script(NULL),
    m_ocontext(ocontext),
    m_obnd(obnd),
    m_base(strdup(base)), 
    m_nb(obnd->getNumBnd()), 
    m_out(NULL)
{
    init(); 
}

void interpolationTest::init()
{
    if(m_interpol)
    {
        m_name = "interpolationTest_interpol.npy" ;
        m_script = "interpolationTest_interpol.py" ;
        m_nx = 820 - 60 + 1 ;     // 761 : 1 nm steps 
        m_ny = m_obnd->getHeight();  // total number of float4 props
    }
    else  // identity 
    {
        m_name = "interpolationTest_identity.npy" ;
        m_script = "interpolationTest_identity.py" ;
        m_nx = m_obnd->getWidth();   // number of wavelength samples
        m_ny = m_obnd->getHeight();  // total number of float4 props
    } 
    LOG(info) << desc(); 
}


std::string interpolationTest::desc() const 
{
    std::stringstream ss ; 
    ss << " name " << m_name 
       << " base " << m_base
       << " script " << m_script
       << " nb " << std::setw(5) << m_nb 
       << " nx " << std::setw(5) << m_nx 
       << " ny " << std::setw(5) << m_ny 
       << " progname " << std::setw(30) << m_progname
       ; 
    return ss.str(); 
}


/**
Hmm is this finding the rearranged PTX locations ?

tests/cu/interpolationTest.cu
**/

void interpolationTest::launch(optix::Context& context)
{
    optix::Buffer buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, m_nx, m_ny);
    context["out_buffer"]->setBuffer(buffer);   

    // rayleighTest uses standard PTX (for loading geometry, materials etc..)
    // as well as test PTX. Hence need to shift the PTX sourcing. 

    OConfig* cfg = m_ocontext->getConfig();
    cfg->setCMakeTarget("interpolationTest");
    cfg->setPTXRel("tests");

    OLaunchTest olt(m_ocontext, m_ok, "interpolationTest.cu", m_progname, "exception");

    olt.setWidth(  m_nx );   
    olt.setHeight( m_ny/8 );   // kernel loops over eight for matsur and num_float4
    olt.launch();

    LOG(info) << olt.brief() ;  


    m_out = NPY<float>::make(m_nx, m_ny, 4);
    m_out->read( buffer->map() );
    buffer->unmap(); 

    LOG(info) 
        << " save "
        << " base " << m_base  
        << " name " << m_name
        ;

    m_out->save(m_base, m_name);
}


int interpolationTest::ana()
{
    std::string path = SSys::Which(m_script);
    LOG(info) 
         << " m_script " << m_script
         << " path " << path 
         ; 

    int RC = path.empty() ? 101 : SSys::exec("python",path.c_str()) ;
    LOG(info) << " RC " << RC ; 
    return RC ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);    

    Opticks ok(argc, argv);
    OpticksHub hub(&ok);
    OScene sc(&hub);

    LOG(info) << " ok " ; 

    OContext* ocontext = sc.getOContext();
    optix::Context context = ocontext->getContext();

    const char* base = "$TMP/optixrap/interpolationTest"  ; 

    OBndLib* obnd = sc.getOBndLib();

    GBndLib* blib = obnd->getBndLib();
    blib->saveAllOverride(base);     

    // Not appropriate to write into geocache from a test
    // BUT ok to override write into $TMP, allowing 
    // python analysis to pick up the potentially dynamic boundary names

    interpolationTest tst(&ok , ocontext, obnd, base);
    tst.launch(context);
    return tst.ana();

}

