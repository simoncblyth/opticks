/**
CSGOptiXRender
=================

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
#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "scuda.h"

#include "Opticks.hh"

#include "CSGFoundry.h"
#include "CSGOptiX.h"


struct CSGOptiXRenderTest
{
    static Opticks* InitOpticks(int argc, char** argv);  

    CSGOptiXRenderTest(int argc, char** argv ) ; 

    Opticks*    ok ; 
    const char* solid_label ; 
    std::vector<unsigned>& solid_selection ; 

    CSGFoundry* fd ; 
    CSGOptiX*   cx ; 

    const char* topline ; 
    const char* botline ; 
    bool        flight ; 
    float4      ce ; 
    const char* default_arg ; 
    std::vector<std::string> args ; 


    void initFD(); 
    void initCX(); 
    void initArgs(); 

    void setCE(const char* arg); 
    void setCE_sla(); 
    void render_snap(const char* namestem);
};


CSGOptiXRenderTest::CSGOptiXRenderTest(int argc, char** argv)
    : 
    ok(InitOpticks(argc, argv)),
    solid_label(ok->getSolidLabel()),         // --solid_label   used for selecting solids from the geometry 
    solid_selection(ok->getSolidSelection()), //  NB its not set yet, that happens below 
    fd(nullptr),
    cx(nullptr),
    topline(SSys::getenvvar("TOPLINE", "CSGOptiXRenderTest")),
    botline(SSys::getenvvar("BOTLINE", nullptr )),
    flight(ok->hasArg("--flightconfig")),
    ce(make_float4(0.f, 0.f, 0.f, 1000.f )), 
    default_arg(SSys::getenvvar("MOI", "sWorld:0:0"))  
{
    initFD(); 
    initCX(); 
    initArgs(); 
}

Opticks* CSGOptiXRenderTest::InitOpticks(int argc, char** argv )
{
    bool has_cfbase = SSys::hasenvvar("CFBASE") ;
    if( has_cfbase )  // when CFBASE override envvvar exists then there is no need for OPTICKS_KEY
    {
        unsetenv("OPTICKS_KEY"); 
    }

    Opticks* ok = new Opticks(argc, argv, has_cfbase ? "--allownokey" : nullptr  );  
    ok->configure(); 
    ok->setRaygenMode(0);  // override --raygenmode option 

    int optix_version_override = CSGOptiX::_OPTIX_VERSION(); 
    const char* out_prefix = ok->getOutPrefix(optix_version_override);  // out_prefix includes values of envvars OPTICKS_GEOM and OPTICKS_RELDIR when defined
    const char* cfbase = ok->getFoundryBase("CFBASE") ; 

    LOG(info) 
        << " optix_version_override " << optix_version_override
        << " out_prefix " << out_prefix
        << " cfbase " << cfbase
        ;

    int create_dirs = 2 ;  
    const char* default_outdir = SPath::Resolve(cfbase, "CSGOptiXRenderTest", out_prefix, create_dirs );  
    const char* outdir = SSys::getenvvar("OPTICKS_OUTDIR", default_outdir );  
    LOG(info) << " default_outdir " << default_outdir ; 
    LOG(info) << " outdir " << outdir ; 

    ok->setOutDir(outdir); 
    ok->writeOutputDirScript(outdir) ; // writes CSGOptiXRenderTest_OUTPUT_DIR.sh in PWD 

    const char* outdir2 = ok->getOutDir(); 
    assert( strcmp(outdir2, outdir) == 0 ); 
    return ok ; 
}

void CSGOptiXRenderTest::initFD()
{
    const char* cfbase = ok->getFoundryBase("CFBASE") ; 
    fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    fd->upload(); 

    if( solid_label )
    {
        fd->findSolidIdx(solid_selection, solid_label); 
        std::string solsel = fd->descSolidIdx(solid_selection); 
        LOG(error) 
            << " --solid_label " << solid_label
            << " solid_selection.size  " << solid_selection.size() 
            << " solid_selection " << solsel 
            ;
    }

    LOG(info) << "foundry " << fd->desc() ; 
    //fd->summary(); 
}

void CSGOptiXRenderTest::initCX()
{
    const char* top    = SSys::getenvvar("TOP", "i0" ); 
    cx = new CSGOptiX(ok, fd ); 
    cx->setTop(top); 

    if( cx->raygenmode > 0 )
    {
        LOG(fatal) << " WRONG EXECUTABLE FOR CSGOptiX::simulate cx.raygenmode " << cx->raygenmode ; 
        assert(0); 
    }
}

void CSGOptiXRenderTest::initArgs()
{
    unsigned num_select = solid_selection.size();  
    if( solid_label )
    {
        assert( num_select > 0 ); 
    }

    const std::vector<std::string>& arglist = ok->getArgList() ;  // --arglist /path/to/arglist.txt

    if( arglist.size() > 0 )
    {    
        std::copy(arglist.begin(), arglist.end(), std::back_inserter(args));
    }
    else
    {
        args.push_back(default_arg); 
    }

    LOG(info) 
        << " arglist.size " << arglist.size()
        << " args.size " << args.size()
        ;
}


/**
CSGOptiXRenderTest::setCE
---------------------------

HMM: solid selection leads to creation of an IAS referencing each of the 
     selected solids so for generality should be using IAS targetting 

**/
void CSGOptiXRenderTest::setCE(const char* arg)
{
    int midx, mord, iidx ;  // mesh-index, mesh-ordinal, instance-index
    fd->parseMOI(midx, mord, iidx,  arg );  
    int rc = fd->getCenterExtent(ce, midx, mord, iidx) ;
    assert(rc); 

    LOG(info) 
        << " arg " << arg 
        << " midx " << midx << " mord " << mord << " iidx " << iidx 
        << " rc " << rc
        << " ce (" << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ") " 
        ; 

    cx->setCE(ce);   // establish the coordinate system 
}

void CSGOptiXRenderTest::setCE_sla()
{
    assert( solid_label ); 
    fd->gasCE(ce, solid_selection );    

    LOG(info) 
        << " solid_label " << solid_label
        << " solid_selection.size " << solid_selection.size()
        << " ce (" << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ") " 
       ; 

    cx->setCE(ce);   // establish the coordinate system 
}

void CSGOptiXRenderTest::render_snap(const char* namestem)
{
    double dt = cx->render();  
    const char* outpath = ok->getOutPath(namestem, ".jpg", -1 ); 

    LOG(error)  
          << " namestem " << namestem 
          << " outpath " << outpath 
          << " dt " << dt 
          ; 

    std::string bottom_line = CSGOptiX::Annotation(dt, botline ); 
    cx->snap(outpath, bottom_line.c_str(), topline  );   
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGOptiXRenderTest t(argc, argv); 

    if( t.solid_label )
    {
        const char* arg = SSys::getenvvar("NAMESTEM", "") ; 
        LOG(info) << " t.solid_label " << t.solid_label << " arg " << arg ; 
        t.setCE_sla(); 
        t.render_snap(arg); 
    }
    else if( t.flight )
    {
        const std::string& _arg = t.args[0];
        const char* arg = _arg.c_str(); 
        LOG(info) << " t.flight arg " << arg  ; 
        t.setCE(arg); 
        t.cx->render_flightpath(); 
    }
    else
    {
        LOG(info) << " t.args.size " << t.args.size()  ; 
        for(unsigned i=0 ; i < t.args.size() ; i++)
        {
            const std::string& _arg = t.args[i];
            const char* arg = _arg.c_str(); 
            t.setCE(arg);
            t.render_snap(arg); 
        }
    }
    return 0 ; 
}


