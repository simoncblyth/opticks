// while CX4GDMLTest ; do sleep 0.2 ; done 

#include <cassert>
// cfg4--;op --cgdmldetector --dbg

#include "CFG4_BODY.hh"

#include "Opticks.hh"     // okc-
#include "OpticksQuery.hh"
#include "OpticksCfg.hh"

#include "OpticksHub.hh"   // okg-

// cfg4-
#include "CTestDetector.hh"
#include "CGDMLDetector.hh"
#include "CMaterialTable.hh"
#include "CBorderSurfaceTable.hh"

// g4-
#include "G4VPhysicalVolume.hh"

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"


// x4-
#include "X4PhysicalVolume.hh"


#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    LOG(info) << argv[0] ;
    Opticks ok(argc, argv);

    OpticksHub hub(&ok);


    OpticksQuery* query = ok.getQuery();   // non-done inside Detector classes for transparent control/flexibility 

    CGDMLDetector* detector  = new CGDMLDetector(&hub, query) ; 

    ok.setIdPathOverride("$TMP");
    detector->saveBuffers();
    ok.setIdPathOverride(NULL);

    bool valid = detector->isValid();
    if(!valid)
    {
        LOG(error) << "CGDMLDetector not valid " ;
        return 0 ;  
    }

    detector->setVerbosity(2) ;

    G4VPhysicalVolume* world_pv = detector->Construct();
    assert(world_pv);

    LOG(info) << "/////////////////////////////////////////////////" ;



   
    GGeo* ggeo = X4PhysicalVolume::Convert(world_pv) ;
    assert( ggeo ); 


    return 0 ; 
}
