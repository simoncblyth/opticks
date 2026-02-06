/**
CSGOptiXRenderTest
====================

For a simpler alternative see CSGOptiXRMTest.cc

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

#include "ssys.h"

#include "SSim.hh"
#include "SOpticks.hh"
#include "SEventConfig.hh"
#include "SBitSet.h"
#include "SGeoConfig.hh"

#include "OPTICKS_LOG.hh"
#include "scuda.h"
#include "sqat4.h"

#include "SGLM.h"
#include "CSGFoundry.h"
#include "CSGCopy.h"
#include "CSGOptiX.h"


const plog::Severity LEVEL = debug ;


struct CSGOptiXRenderTest
{
    CSGOptiXRenderTest() ;

    const char* solid_selection ;

    SGLM*       gm ; 
    CSGFoundry* fd ;  // possibly with selection applied
    CSGOptiX*   cx ;

    const char* flight ;
    float4      ce ;

    const char* default_arg ;
    std::vector<std::string> args ;

    void init();
    void initGeom();
    void initSolidSelection();
    void initArgs();

    void setFrame_solid_selection();

};


CSGOptiXRenderTest::CSGOptiXRenderTest()
    :
    solid_selection(SGeoConfig::SolidSelection()),   //  formerly from --solid_label option used for selecting solids from the geometry
    gm(SGLM::Get()),
    fd(CSGFoundry::Load()),
    cx(CSGOptiX::Create(fd)),   // uploads fd and then instanciates
    flight(SGeoConfig::FlightConfig()),
    ce(make_float4(0.f, 0.f, 0.f, 1000.f )),
    default_arg(ssys::getenvvar("MOI", "sWorld:0:0"))
{
    init();
}


void CSGOptiXRenderTest::init()
{
    initGeom();
    initSolidSelection();
    initArgs();
}

void CSGOptiXRenderTest::initGeom()
{
    stree* tree = fd->getTree();
    SScene* scene = fd->getScene() ;
    gm->setTreeScene(tree, scene);
    gm->set_frame();   // MOI frame initially
}




/**
CSGOptiXRenderTest::initSolidSelection
---------------------------------------

MAYBE : remove this solid selection approach, as EMM does similar but more cleanly ?

**/

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
        bool num_select_expect = num_select > 0 ;
        assert( num_select_expect );
        if(!num_select_expect) std::raise(SIGINT);
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

void CSGOptiXRenderTest::setFrame_solid_selection()
{
    assert( solid_selection );
    fd->gasCE(ce, cx->solid_selection );

    LOG(LEVEL)
        << " solid_selection " << solid_selection
        << " cx.solid_selection.size " << cx->solid_selection.size()
        << " ce (" << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ") "
       ;

    gm->set_frame(ce);
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    SEventConfig::SetRGModeRender();
    SSim::Create();

    CSGOptiXRenderTest t;
    CSGOptiX* cx = t.cx ;
    SGLM* gm = t.gm ;

    if( t.solid_selection )
    {
        const char* arg = ssys::getenvvar("NAMESTEM", "") ;
        LOG(LEVEL) << " t.solid_selection " << t.solid_selection << " arg " << arg ;
        t.setFrame_solid_selection();
        cx->render();
    }
    else if( t.flight )
    {
        const char* arg = t.args[0].c_str();
        LOG(LEVEL) << " t.flight arg " << arg  ;
        gm->set_frame(arg);
        cx->render_flightpath();
    }
    else
    {
        //LOG(LEVEL) << " t.args.size " << t.args.size()  ;
        for(unsigned i=0 ; i < t.args.size() ; i++)
        {
            const char* arg = t.args[i].c_str();
            //LOG(LEVEL) << " arg:" << ( arg ? arg : "-" ) ;
            gm->set_frame(arg);
            cx->render();
        }
    }
    return 0 ;
}

