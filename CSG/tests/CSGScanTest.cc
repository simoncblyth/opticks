// ./CSGScanTest.sh

#include <vector>
#include <cassert>
#include <iostream>

#include "scuda.h"

#include "CSGFoundry.h"
#include "CSGSolid.h"
#include "CSGScan.h"


int main(int argc, char** argv)
{
    const char* dir = "/tmp/CSGScanTest_scans" ; 
    CSGFoundry fd ;  

    //fd.makeDemoSolids(); 
    fd.makeEllipsoid(); 
    //const char* name = "sphe" ; 
    //fd.makeClustered(name, 0,1,1, 0,1,1,  0,2,1, 100. );  

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
