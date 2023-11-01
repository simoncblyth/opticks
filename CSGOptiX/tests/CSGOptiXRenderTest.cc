/**
CSGOptiXRenderTest
====================

For a simpler alternative see CSGOptiXRdrTest

This executable is typically run via cxr_*.sh specialized bash scripts 
each dedicated to different types of renders. 
The scripts control what this executable does and where it writes via envvars.  
See CSGOptiX/index.rst for details on the scripts. 

With option --arglist /path/to/arglist.txt each line of the arglist file 
is taken as an MOI specifying the center_extent box to target. 
Without an --arglist option the MOI envvar or default value  "sWorld:0:0" 
is used to set the viewpoint target box.

important envvars

MOI 
    specifies the viewpoint target box, default "sWorld:0:0" 
TOP
    selects the part of the geometry to use, default i0 
CFBASE
    directory to load the CSGFoundry geometry from, default "$TMP/CSG_GGeo" 

    NB CFBASE is now only used as an override (eg for demo geometry) 
    when not rendering the standard OPTICKS_KEY geometry which is now located inside geocache.
**/

#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <csignal>

#include "scontext.h"

#include "SPath.hh"
#include "SStr.hh"
#include "SSys.hh"
#include "SSim.hh"
#include "SOpticks.hh"
#include "SEventConfig.hh"
#include "SOpticksResource.hh"
#include "SBitSet.hh"
#include "SGeoConfig.hh"

#include "OPTICKS_LOG.hh"
#include "scuda.h"
#include "sqat4.h"

#include "SGLM.h"

#ifdef WITH_SGLM
#else
#include "Opticks.hh"
#endif

#include "CSGFoundry.h"
#include "CSGCopy.h"
#include "CSGOptiX.h"


const plog::Severity LEVEL = debug ; 


struct CSGOptiXRenderTest
{
    CSGOptiXRenderTest() ; 

#ifdef WITH_SGLM
#else
    Opticks*    ok ; 
#endif

    const char* solid_selection ; 

    CSGFoundry* fd ;  // possibly with selection applied
    CSGOptiX*   cx ; 

    const char* flight ; 
    float4      ce ; 

    const char* default_arg ; 
    std::vector<std::string> args ; 

    void initSolidSelection(); 
    void initArgs(); 

    void setFrame_sla(); 

};


CSGOptiXRenderTest::CSGOptiXRenderTest()
    : 
#ifdef WITH_SGLM
#else
    ok(Opticks::Get()),
#endif
    solid_selection(SGeoConfig::SolidSelection()),   //  formerly from --solid_label option used for selecting solids from the geometry 
    fd(CSGFoundry::Load()), 
    cx(CSGOptiX::Create(fd)),   // uploads fd and then instanciates 
    flight(SGeoConfig::FlightConfig()),
    ce(make_float4(0.f, 0.f, 0.f, 1000.f )),
    default_arg(SSys::getenvvar("MOI", "sWorld:0:0"))  
{
    initSolidSelection(); 
    initArgs(); 
}


void CSGOptiXRenderTest::initSolidSelection()
{
    if( solid_selection == nullptr  ) return ; 

    fd->findSolidIdx(cx->solid_selection, solid_selection); 
    std::string solsel = fd->descSolidIdx(cx->solid_selection); 
    LOG(error) 
        << " solid_selection " << solid_selection
        << " cx.solid_selection.size  " << cx->solid_selection.size() 
        << " solsel " << solsel 
        ;

}

void CSGOptiXRenderTest::initArgs()
{
    unsigned num_select = cx->solid_selection.size();  
    if( solid_selection )
    {
        assert( num_select > 0 ); 
    }

    std::vector<std::string>* arglist = SGeoConfig::Arglist() ;

    if( arglist && arglist->size() > 0 )
    {    
        std::copy(arglist->begin(), arglist->end(), std::back_inserter(args));
        LOG(LEVEL) << " using arglist from SGeoConfig::Arglist " ; 
    }
    else
    {
        args.push_back(default_arg);   
        //LOG(LEVEL) << " using default_arg from MOI envvar " ; 
    }
}

void CSGOptiXRenderTest::setFrame_sla()
{
    assert( solid_selection ); 
    fd->gasCE(ce, cx->solid_selection );    

    LOG(LEVEL) 
        << " solid_selection " << solid_selection
        << " cx.solid_selection.size " << cx->solid_selection.size()
        << " ce (" << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ") " 
       ; 

    cx->setFrame(ce); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEventConfig::SetRGMode("render"); 

#ifdef WITH_SGLM
    //LOG(LEVEL) << " WITH_SGLM : not using Opticks " ; 
#else
    LOG(fatal) << " not-WITH_SGLM : calling Opticks::Configure  " ; 
    Opticks::Configure(argc, argv);  
#endif

    const char* outdir = SEventConfig::OutDir(); 
    SOpticks::WriteOutputDirScript(outdir) ; // writes CSGOptiXRenderTest_OUTPUT_DIR.sh in PWD 


    SSim::Create(); 

    CSGOptiXRenderTest t; 

    if( t.solid_selection )
    {
        const char* arg = SSys::getenvvar("NAMESTEM", "") ; 
        LOG(LEVEL) << " t.solid_selection " << t.solid_selection << " arg " << arg ; 
        t.setFrame_sla(); 
        t.cx->render(); 
    }
    else if( t.flight )
    {
        const char* arg = t.args[0].c_str(); 
        LOG(LEVEL) << " t.flight arg " << arg  ; 
        t.cx->setFrame(arg); 
        t.cx->render_flightpath(); 
    }
    else
    {
        //LOG(LEVEL) << " t.args.size " << t.args.size()  ; 
        for(unsigned i=0 ; i < t.args.size() ; i++)
        {
            const char* arg = t.args[i].c_str(); 
            //LOG(LEVEL) << " arg:" << ( arg ? arg : "-" ) ; 
            t.cx->setFrame(arg); 
            t.cx->render(); 
        }
    }
    return 0 ; 
}

