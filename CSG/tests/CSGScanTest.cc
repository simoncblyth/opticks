// ~/o/CSG/tests/CSGScanTest.sh

#include "OPTICKS_LOG.hh"

#include <vector>
#include <cassert>
#include <iostream>

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

    const char* dir_default = ssys::getenvvar("CSGSCANTEST_BASE", "$TMP/CSGScanTest_scans" ); 
    const char* dir = spath::Resolve(dir_default) ; 
    LOG(info) << " CSGSCANTEST_BASE dir " << dir ; 

    const char* solid = ssys::getenvvar("CSGSCANTEST_SOLID", "Ellipsoid" ); 
    LOG(info) << " CSGSCANTEST_SOLID " << solid ; 

    SSim::Create(); 

    CSGFoundry fd ;  
    fd.maker->make( solid ); 

    unsigned numSolid = fd.getNumSolid() ; 
    LOG(info) << "[ numSolid " << numSolid  ; 

    for(unsigned i=0 ; i < numSolid ; i++)
    {
        const CSGSolid* solid = fd.getSolid(i); 

        CSGScan sc(dir, &fd, solid); 
        sc.axis_scan(); 
        sc.rectangle_scan(); 
        sc.circle_scan(); 
    }


    LOG(info) << "] numSolid " << numSolid  ; 

    return 0 ;  
}
