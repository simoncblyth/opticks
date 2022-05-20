/**
CSGOptiXRenderTest
====================

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

#include "SPath.hh"
#include "SStr.hh"
#include "SSys.hh"
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

struct CSGOptiXRenderTest
{
    CSGOptiXRenderTest() ; 

#ifdef WITH_SGLM
    static constexpr const char* VARIANT = "SGLM" ;    
#else
    static constexpr const char* VARIANT = "Comp" ;    
    Opticks*    ok ; 
#endif

    const char* solid_selection ; 

    CSGFoundry* fdl    ; // as loaded
    const SBitSet* elv ; 

    CSGFoundry* fd ;  // possibly with selection applied
    CSGOptiX*   cx ; 

    const char* flight ; 

    float4      ce ; 
    qat4*       m2w ; 
    qat4*       w2m ; 

    const char* default_arg ; 
    std::vector<std::string> args ; 

    void initCX(); 
    void initSS(); 
    void initArgs(); 

    void setComposition(const char* moi); 
    void setComposition_sla(); 

    void render_snap(const char* namestem);
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
    m2w(qat4::identity()),
    w2m(qat4::identity()),
    default_arg(SSys::getenvvar("MOI", "sWorld:0:0"))  
{
    initCX(); 
    initSS(); 
    initArgs(); 
}

void CSGOptiXRenderTest::initCX()
{
    assert(cx); 
    bool expect =  cx->raygenmode == 0 ;  
    if(!expect) LOG(fatal) << " WRONG EXECUTABLE FOR CSGOptiX::simulate cx.raygenmode " << cx->raygenmode ; 
    assert(expect); 
}

void CSGOptiXRenderTest::initSS()
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
    }
    else
    {
        args.push_back(default_arg);   // default_arg is value of MOI envvar 
    }

    LOG(info) 
        << " default_arg " << default_arg
        << " arglist->size " << ( arglist ? arglist->size() : 0 ) 
        << " args.size " << args.size()
        ;
}

void CSGOptiXRenderTest::setComposition_sla()
{
    assert( solid_selection ); 
    fd->gasCE(ce, cx->solid_selection );    

    LOG(info) 
        << " solid_selection " << solid_selection
        << " cx.solid_selection.size " << cx->solid_selection.size()
        << " ce (" << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ") " 
       ; 

    cx->setComposition(ce, nullptr, nullptr);   // establish the coordinate system 
}



/**
CSGOptiXRenderTest::setComposition
------------------------------------

HMM: solid selection leads to creation of an IAS referencing each of the 
     selected solids so for generality should be using IAS targetting 

For global geometry which typically uses default iidx of 0 there is special 
handling of iidx -1/-2/-3 implemented in CSGTarget::getCenterExtent


iidx -2
    ordinary xyzw frame calulated by SCenterExtentFrame

iidx -3
    rtp tangential frame calulated by SCenterExtentFrame

**/
void CSGOptiXRenderTest::setComposition(const char* moi)
{
    cx->setComposition(moi);  
}



void CSGOptiXRenderTest::render_snap(const char* namestem)
{
    std::string name = SStr::Format("%s_%s", VARIANT, namestem ); 
    cx->render_snap(name.c_str()); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    SEventConfig::SetRGMode("render"); 

    LOG(info) << " getenv.CAM " << getenv("CAM") ; 
    LOG(info) << " getenv.CAMERATYPE " << getenv("CAMERATYPE") ; 

#ifdef WITH_SGLM
#else
    Opticks::Configure(argc, argv);  
#endif

    const char* outdir = SEventConfig::OutDir(); 
    SOpticks::WriteOutputDirScript(outdir) ; // writes CSGOptiXRenderTest_OUTPUT_DIR.sh in PWD 


    CSGOptiXRenderTest t; 

    if( t.solid_selection )
    {
        const char* arg = SSys::getenvvar("NAMESTEM", "") ; 
        LOG(info) << " t.solid_selection " << t.solid_selection << " arg " << arg ; 
        t.setComposition_sla(); 
        t.render_snap(arg); 
    }
    else if( t.flight )
    {
        const char* arg = t.args[0].c_str(); 
        LOG(info) << " t.flight arg " << arg  ; 
        t.setComposition(arg); 
        t.cx->render_flightpath(); 
    }
    else
    {
        LOG(info) << " t.args.size " << t.args.size()  ; 
        for(unsigned i=0 ; i < t.args.size() ; i++)
        {
            const char* arg = t.args[i].c_str(); 
            t.setComposition(arg); 
            t.render_snap(arg); 
        }
    }
    return 0 ; 
}

