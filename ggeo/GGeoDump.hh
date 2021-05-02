#pragma once

#include "GGEO_API_EXPORT.hh"
 
class GGeo ; 
class GGeoLib ; 

struct GGEO_API GGeoDump
{
    const GGeo*    ggeo ; 
    const GGeoLib* geolib ; 

    GGeoDump(const GGeo* ggeo); 

    void dump(int repeatIdx, int primIdx, int partIdxRel );
    void dump_(unsigned repeatIdx);
    void dump_(unsigned repeatIdx, unsigned primIdx );
    void dump_(unsigned repeatIdx, unsigned primIdx, unsigned partIdxRel );
};




