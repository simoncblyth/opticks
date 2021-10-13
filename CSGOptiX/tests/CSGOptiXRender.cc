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


**/

#include <algorithm>
#include <iterator>

#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "sutil_vec_math.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"


int main(int argc, char** argv)
{
    for(int i=0 ; i < argc ; i++ ) std::cout << argv[i] << std::endl; 

    OPTICKS_LOG(argc, argv); 

    bool overwrite = false ;  // allows commandline and envvars to override this default setting, see Opticks::getOutDir
    SSys::setenvvar("OPTICKS_OUTDIR", "$TMP/CSGOptiX", overwrite );

    Opticks ok(argc, argv); 
    ok.configure(); 
    ok.setRaygenMode(0);             // override --raygenmode option 

    const char* top    = SSys::getenvvar("TOP", "i0" ); 
    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" );


    const char* solid_label = ok.getSolidLabel();  // --solid_label   used for selecting solids from the geometry 
    std::vector<unsigned>& solid_selection = ok.getSolidSelection(); 

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
