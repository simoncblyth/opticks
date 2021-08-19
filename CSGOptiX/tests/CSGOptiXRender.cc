/**
CSGOptiXRender
=================

With option --arglist /path/to/arglist.txt each line of the arglist file 
is taken as an MOI specifying the center_extent box to target. 
Without an --arglist option the MOI envvar or default value  "sWorld:0:0" 
is consulted to set the target box.
 

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
    Opticks ok(argc, argv); 
    ok.configure(); 
    ok.setRaygenMode(0); // override --raygenmode option 

    const char* top    = SSys::getenvvar("TOP", "i0" ); 
    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" );
    const char* solid_label = ok.getSolidLabel(); 
    std::vector<unsigned>& solid_selection = ok.getSolidSelection(); 

    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    fd->upload(); 

    if( solid_label )
    {
        fd->findSolidIdx(solid_selection, solid_label); 

        std::stringstream ss ; 
        ss << "(" ; 
        for(int i=0 ; i < int(solid_selection.size()) ; i++) ss << solid_selection[i] << " " ; 
        ss << ")" ; 
        std::string solsel = ss.str() ; 

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
            int midx, mord, iidx ; 
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
                const char* path = ok.getOutPath(namestem, ext, index ); 
                LOG(error) << " path " << path ; 

                std::string bottom_line = CSGOptiX::Annotation(dt, botline ); 
                cx.snap(path, bottom_line.c_str(), topline  );   
            }
        }
        else
        {
            LOG(error) << " SKIPPING as failed to lookup CE " << arg ; 
        }
    }
    return 0 ; 
}
