// ~/o/CSG/tests/CSGScanTest.sh

#include "OPTICKS_LOG.hh"


#include "scuda.h"
#include "SSim.hh"
#include "ssys.h"
#include "spath.h"

#include "CSGFoundry.h"
#include "CSGMaker.h"
#include "CSGSolid.h"
#include "CSGScan.h"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);    

    const char* geom = ssys::getenvvar("GEOM", "JustOrb" );
    LOG(info) << " GEOM " << geom ; 

    SSim::Create(); 

    CSGFoundry fd ;  
    fd.maker->make( geom ); 
    fd.upload(); 

    const CSGSolid* solid = fd.getSolid(0); 
    // TODO: pick solid/prim from full geometry : not just trivial ones

    CSGScan sc( &fd, solid, "axis,rectangle,circle" ); 
    sc.intersect_h(); 
    sc.intersect_d();

    std::cout << sc.brief() ; 
 
    sc.save("$FOLD", geom); 


    return 0 ;  
}
