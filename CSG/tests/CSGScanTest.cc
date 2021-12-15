// ./CSGScanTest.sh

#include "OPTICKS_LOG.hh"

#include <vector>
#include <cassert>
#include <iostream>

#include "scuda.h"
#include "SSys.hh"
#include "SPath.hh"

#include "CSGFoundry.h"
#include "CSGMaker.h"
#include "CSGSolid.h"
#include "CSGScan.h"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);    

    int create_dirs = 2 ; // 2: dirpath
    const char* dir_default = SSys::getenvvar("CSGSCANTEST_BASE", "$TMP/CSGScanTest_scans" ); 
    const char* dir = SPath::Resolve(dir_default, create_dirs) ; 
    LOG(info) << " CSGSCANTEST_BASE dir " << dir ; 

    const char* solid = SSys::getenvvar("CSGSCANTEST_SOLID", "elli" ); 
    LOG(info) << " CSGSCANTEST_SOLID " << solid ; 

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
