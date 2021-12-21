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


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    bool override_cfbase =  SSys::hasenvvar("CFBASE") ; 
    const char* argforce = override_cfbase ? "--allownokey" : nullptr  ; 
    if( override_cfbase )
    {
        unsetenv("OPTICKS_KEY"); 
    }
    // placing CFBASE inside idpath means that in default situation with standard geocache geometry 
    // cannot --allownokey but when the CFBASE override envvvar exists then there is 
    // no need for OPTICKS_KEY  

    Opticks ok(argc, argv, argforce ); 
    ok.configure(); 
    ok.setRaygenMode(0);             // override --raygenmode option 

    int optix_version_override = CSGOptiX::_OPTIX_VERSION(); 
    const char* out_prefix = ok.getOutPrefix(optix_version_override);   
    // out_prefix includes values of envvars OPTICKS_GEOM and OPTICKS_RELDIR when defined
    LOG(info) 
        << " optix_version_override " << optix_version_override
        << " out_prefix " << out_prefix
        ;

    const char* top    = SSys::getenvvar("TOP", "i0" ); 
    const char* cfbase = ok.getFoundryBase("CFBASE") ; 

    int create_dirs = 2 ;  
    const char* default_outdir = SPath::Resolve(cfbase, "CSGOptiXRenderTest", out_prefix, create_dirs );  
    const char* outdir = SSys::getenvvar("OPTICKS_OUTDIR", default_outdir );  
    LOG(info) << " default_outdir " << default_outdir ; 
    LOG(info) << " outdir " << outdir ; 

    ok.setOutDir(outdir); 

    const char* outdir2 = ok.getOutDir(); 
    assert( strcmp(outdir2, outdir) == 0 ); 


    const char* solid_label = ok.getSolidLabel();  // --solid_label   used for selecting solids from the geometry 
    std::vector<unsigned>& solid_selection = ok.getSolidSelection(); // NB its not set yet, that happens below 

    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
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
    unsigned num_select = solid_selection.size();  


    LOG(info) << "foundry " << fd->desc() ; 
    //fd->summary(); 

    CSGOptiX cx(&ok, fd); 
    cx.setTop(top); 

    if( cx.raygenmode > 0 )
    {
        LOG(fatal) << " WRONG EXECUTABLE FOR CSGOptiX::simulate cx.raygenmode " << cx.raygenmode ; 
        assert(0); 
    }

    bool flight = ok.hasArg("--flightconfig") ; 
    const std::vector<std::string>& arglist = ok.getArgList() ;  // --arglist /path/to/arglist.txt
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

    for(unsigned i=0 ; i < args.size() ; i++)
    {
        const std::string& arg = args[i];
        const char* namestem = num_select == 0 ? arg.c_str() : SSys::getenvvar("NAMESTEM", "")  ; 

        int rc = 0 ; 
        float4 ce = make_float4(0.f, 0.f, 0.f, 1000.f ); 
        if(solid_selection.size() > 0)
        {   
            // HMM: solid selection leads to creation of an IAS referencing each of the 
            //      selected solids so for generality should be using IAS targetting 
            //
            fd->gasCE(ce, solid_selection );    
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
            cx.setCE(ce);   // establish the coordinate system 

            if(flight)
            {
                cx.render_flightpath(); 
            }
            else
            {
                double dt = cx.render();  

                const char* ext = ".jpg" ; 
                int index = -1 ;  
                const char* outpath = ok.getOutPath(namestem, ext, index ); 
                LOG(error) << " outpath " << outpath ; 

                std::string bottom_line = CSGOptiX::Annotation(dt, botline ); 
                cx.snap(outpath, bottom_line.c_str(), topline  );   
            }
        }
        else
        {
            LOG(error) << " SKIPPING as failed to lookup CE " << arg ; 
        }
    }
    return 0 ; 
}