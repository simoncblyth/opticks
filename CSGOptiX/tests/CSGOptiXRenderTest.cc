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
#include "SSys.hh"
#include "SOpticksResource.hh"
#include "SBitSet.hh"
#include "OPTICKS_LOG.hh"
#include "scuda.h"
#include "sqat4.h"

// TODO: avoid use of Opticks.hh here 
#include "Opticks.hh"

#include "CSGFoundry.h"
#include "CSGCopy.h"
#include "CSGOptiX.h"


struct CSGOptiXRenderTest
{
    static const char* InitCFBASE(); 
    static Opticks* InitOpticks(int argc, char** argv, const char* cfbase );  
    static void InitOutdir(Opticks* ok, const char* cfbase); 


    CSGOptiXRenderTest(int argc, char** argv ) ; 

    const char* cfbase ; 
    Opticks*    ok ; 
    const char* solid_label ; 
    std::vector<unsigned>& solid_selection ; 

    CSGFoundry* fdl    ; // as loaded
    const SBitSet* elv ; 

    CSGFoundry* fd ;  // possibly with selection applied
    CSGOptiX*   cx ; 

    const char* topline ; 
    const char* botline ; 
    bool        flight ; 
    float4      ce ; 
    qat4*       m2w ; 
    qat4*       w2m ; 

    const char* default_arg ; 
    std::vector<std::string> args ; 


    void initFD(const char* cfbase); 
    void initCX(); 
    void initArgs(); 

    void setCE(const char* moi); 
    void setCE_sla(); 
    void render_snap(const char* namestem);
};


CSGOptiXRenderTest::CSGOptiXRenderTest(int argc, char** argv)
    : 
    cfbase(InitCFBASE()),
    ok(InitOpticks(argc, argv, cfbase)),
    solid_label(ok->getSolidLabel()),         // --solid_label   used for selecting solids from the geometry 
    solid_selection(ok->getSolidSelection()), //  NB its not set yet, that happens below 
    fdl(nullptr),
    elv(nullptr),
    fd(nullptr),
    cx(nullptr),
    topline(SSys::getenvvar("TOPLINE", "CSGOptiXRenderTest")),
    botline(SSys::getenvvar("BOTLINE", nullptr )),
    flight(ok->hasArg("--flightconfig")),
    ce(make_float4(0.f, 0.f, 0.f, 1000.f )),
    m2w(qat4::identity()),
    w2m(qat4::identity()),
    default_arg(SSys::getenvvar("MOI", "sWorld:0:0"))  
{
    initFD(cfbase); 
    initCX(); 
    initArgs(); 
}



const char* CSGOptiXRenderTest::InitCFBASE()
{
    bool has_cfbase = SSys::hasenvvar("CFBASE") ;

    if( has_cfbase )  // when CFBASE override envvvar exists then there is no need for OPTICKS_KEY
    {
        LOG(info) << " CFBASE envvar detected : ignoring OPTICKS_KEY " ; 
        unsetenv("OPTICKS_KEY"); 
    }
    else
    {
        LOG(info) << " no CFBASE envvar detected : will need OPTICKS_KEY " ; 
    }

    const char* cfbase = SOpticksResource::CFBase("CFBASE") ;  // sensitive to OPTICKS_KEY 
    return cfbase ; 
}


/**
TODO: eliminate use of Opticks here, its doing little 
**/

Opticks* CSGOptiXRenderTest::InitOpticks(int argc, char** argv, const char* cfbase )
{
    Opticks* ok = new Opticks(argc, argv, cfbase ? "--allownokey" : nullptr  );  
    ok->configure(); 
    ok->setRaygenMode(0);  // override --raygenmode option 
    InitOutdir(ok, cfbase); 
    return ok ; 
}

/**
CSGOptiXRenderTest::InitOutdir
-------------------------------

The out_prefix depends on values of envvars OPTICKS_GEOM and OPTICKS_RELDIR when defined.

TODO: wean off Opticks, move file handling basics down to sysrap 

**/

void CSGOptiXRenderTest::InitOutdir(Opticks* ok, const char* cfbase)
{
    int optix_version_override = CSGOptiX::_OPTIX_VERSION(); 
    const char* out_prefix = ok->getOutPrefix(optix_version_override);

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
}


void CSGOptiXRenderTest::initFD(const char* cfbase)
{
    fdl = CSGFoundry::Load(cfbase, "CSGFoundry"); 

    elv = SBitSet::Create( fdl->getNumMeshName(), "ELV", nullptr ) ;  

    if(elv)
    { 
        LOG(info) << elv->desc() << std::endl << fdl->descELV(elv) ; 
        fd = CSGCopy::Select(fdl, elv );  
    }
    else
    {
        fd = fdl ; 
    }

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
    cx = new CSGOptiX(ok, fd ); 

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
        args.push_back(default_arg);   // default_arg is value of MOI envvar 
    }

    LOG(info) 
        << " default_arg " << default_arg
        << " arglist.size " << arglist.size()
        << " args.size " << args.size()
        ;
}


/**
CSGOptiXRenderTest::setCE
---------------------------

HMM: solid selection leads to creation of an IAS referencing each of the 
     selected solids so for generality should be using IAS targetting 

For global geometry which typically uses default iidx of 0 there is special 
handling of iidx -1/-2/-3 implemented in CSGTarget::getCenterExtent


iidx -2
    ordinary xyzw frame calulated by SCenterExtentFrame

iidx -3
    rtp tangential frame calulated by SCenterExtentFrame


**/
void CSGOptiXRenderTest::setCE(const char* moi)
{
    int midx, mord, iidx ;  // mesh-index, mesh-ordinal, instance-index
    fd->parseMOI(midx, mord, iidx,  moi );  

    int rc = fd->getCenterExtent(ce, midx, mord, iidx, m2w, w2m ) ;

    LOG(info) 
        << " moi " << moi 
        << " midx " << midx << " mord " << mord << " iidx " << iidx 
        << " rc [" << rc << "]" 
        << " ce (" << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ") " 
        << " m2w (" << *m2w << ")"    
        << " w2m (" << *w2m << ")"    
        ; 

    assert(rc==0); 
    cx->setComposition(ce, m2w, w2m );   // establish the coordinate system 
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

    cx->setComposition(ce, nullptr, nullptr);   // establish the coordinate system 
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


