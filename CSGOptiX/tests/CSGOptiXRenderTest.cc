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

    void initFD(); 
    void initCX(); 

};


CSGOptiXRenderTest::CSGOptiXRenderTest(int argc, char** argv)
    : 
    ok(InitOpticks(argc, argv)),
    solid_label(ok->getSolidLabel()),         // --solid_label   used for selecting solids from the geometry 
    solid_selection(ok->getSolidSelection()), //  NB its not set yet, that happens below 
    fd(nullptr),
    cx(nullptr)
{
    initFD(); 
    initCX(); 
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



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGOptiXRenderTest t(argc, argv); 

    Opticks* ok = t.ok ; 
    CSGFoundry* fd = t.fd ; 
    CSGOptiX* cx = t.cx ; 


    bool flight = ok->hasArg("--flightconfig") ; 
    const std::vector<std::string>& arglist = ok->getArgList() ;  // --arglist /path/to/arglist.txt
    const char* topline = SSys::getenvvar("TOPLINE", "CSGOptiXRender") ; 
    const char* botline = SSys::getenvvar("BOTLINE", nullptr ) ; 

    // arglist allows multiple snaps with different viewpoints to be created  
    std::vector<std::string> args ; 
    if( arglist.size() > 0 )
    {    
        std::copy(arglist.begin(), arglist.end(), std::back_inserter(args));
    }
    else
    {
        args.push_back(SSys::getenvvar("MOI", "sWorld:0:0"));  
    }


    LOG(info) << " args.size " << args.size() ; 
    unsigned num_select = t.solid_selection.size();  

    for(unsigned i=0 ; i < args.size() ; i++)
    {
        const std::string& arg = args[i];
        const char* namestem = num_select == 0 ? arg.c_str() : SSys::getenvvar("NAMESTEM", "")  ; 

        LOG(info) 
            << " i " << i 
            << " arg " << arg 
            << " num_select " << num_select 
            << " namestem " << namestem
            ;

        int rc = 0 ; 
        float4 ce = make_float4(0.f, 0.f, 0.f, 1000.f ); 
        if(t.solid_selection.size() > 0)
        {   
            // HMM: solid selection leads to creation of an IAS referencing each of the 
            //      selected solids so for generality should be using IAS targetting 
            //
            fd->gasCE(ce, t.solid_selection );    
        }
        else
        {
            int midx, mord, iidx ;  // mesh-index, mesh-ordinal, instance-index
            fd->parseMOI(midx, mord, iidx,  arg.c_str() );  
            LOG(info) << " i " << i << " arg " << arg << " midx " << midx << " mord " << mord << " iidx " << iidx ;   
            rc = fd->getCenterExtent(ce, midx, mord, iidx) ;
        }

        if(rc == 0 )
        {
            cx->setCE(ce);   // establish the coordinate system 

            if(flight)
            {
                cx->render_flightpath(); 
            }
            else
            {
                double dt = cx->render();  

                const char* ext = ".jpg" ; 
                int index = -1 ;  
                const char* outpath = ok->getOutPath(namestem, ext, index ); 
                LOG(error) << " outpath " << outpath ; 

                std::string bottom_line = CSGOptiX::Annotation(dt, botline ); 
                cx->snap(outpath, bottom_line.c_str(), topline  );   
            }
        }
        else
        {
            LOG(error) << " SKIPPING as failed to lookup CE " << arg ; 
        }
    }
    return 0 ; 
}
