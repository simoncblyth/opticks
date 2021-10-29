// ./CSGScanTest.sh

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
    //const char* dir = "/tmp/CSGScanTest_scans" ; 
    int create_dirs = 2 ; // 2: dirpath

    const char* dir_default = SSys::getenvvar("CSGSCANTEST_BASE", "$TMP/CSGScanTest_scans" ); 
    const char* dir = SPath::Resolve(dir_default, create_dirs) ; 
    std::cout << " dir " << dir << std::endl ; 


    const char* solid = SSys::getenvvar("CSGSCANTEST_SOLID", "elli" ); 

    CSGFoundry fd ;  

    std::cout << "CSGSCANTEST_SOLID " << solid << std::endl; 
    fd.maker->make( solid ); 

    unsigned numSolid = fd.getNumSolid() ; 
    std::cout << "numSolid " << numSolid << std::endl ; 

    for(unsigned i=0 ; i < numSolid ; i++)
    {
        const CSGSolid* solid = fd.getSolid(i); 

        CSGScan sc(dir, &fd, solid); 
        sc.axis_scan(); 
        sc.rectangle_scan(); 
        sc.circle_scan(); 
    }
    return 0 ;  
}
