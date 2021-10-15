#include "CSGOptiXSimulate.h"
#include "PLOG.hh"

void CSGOptiXSimulate::ParseCEGS( uint4& cegs, float4& ce )  // static 
{
    std::vector<int> vcegs ; 
    SSys::getenvintvec("CEGS", vcegs, ':', "5:0:5:1000" ); 

    cegs.x = vcegs.size() > 0 ? vcegs[0] : 5  ; 
    cegs.y = vcegs.size() > 1 ? vcegs[1] : 0  ; 
    cegs.z = vcegs.size() > 2 ? vcegs[2] : 5 ; 
    cegs.w = vcegs.size() > 3 ? vcegs[3] : 1000 ; 

    int4 oce ;  // override ce
    oce.x = vcegs.size() > 4 ? vcegs[4] : 0 ; 
    oce.y = vcegs.size() > 5 ? vcegs[5] : 0 ; 
    oce.z = vcegs.size() > 6 ? vcegs[6] : 0 ; 
    oce.w = vcegs.size() > 7 ? vcegs[7] : 0 ; 

    if( oce.w > 0 )   // require 8 delimited ints to override the MOI.ce
    {
        ce.x = float(oce.x); 
        ce.y = float(oce.y); 
        ce.z = float(oce.z); 
        ce.w = float(oce.w); 
        LOG(info) << "override the MOI.ce with CEGS.ce (" << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ")" ;  
    } 

    LOG(info) 
        << " CEGS nx:ny:nz:photons_per_genstep " << cegs.x << ":" << cegs.y << ":" << cegs.z << ":" << cegs.w 
        ;   

}


