// ./GeoTest.sh

#include <vector>
#include <cassert>
#include <iostream>

#include "scuda.h"
#include "CSGFoundry.h"
#include "CSGSolid.h"

#include "Scan.h"
#include "Geo.h"
#include "Util.h"
#include "View.h"

int main(int argc, char** argv)
{
    CSGFoundry foundry ;  
    Geo geo(&foundry) ; 

    unsigned width = 1280u ; 
    unsigned height = 720u ; 

    glm::vec4 eye_model ; 
    Util::GetEVec(eye_model, "EYE", "-1.0,-1.0,1.0,1.0"); 

    const float4 gce = geo.getCenterExtent() ;   
    glm::vec4 ce(gce.x,gce.y,gce.z, gce.w*1.4f );   // defines the center-extent of the region to view

    View view = {} ; 
    view.update(eye_model, ce, width, height) ; 
    view.dump("View::dump"); 
    view.save("/tmp");  

    const char* dir = "/tmp/GeoTest_scans" ; 

    const CSGSolid* solid0 = foundry.getSolid(0); 

    Scan scan(dir, &foundry, solid0 ); 
    scan.axis_scan() ; 
    scan.rectangle_scan() ; 
    scan.circle_scan() ; 

    return 0 ;  
}
