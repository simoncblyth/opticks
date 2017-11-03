#pragma once

#include "GGEO_API_EXPORT.hh"

class GSourceLib ; 
class GScintillatorLib ; 
class GBndLib ; 
class GGeoLib ; 
class GPmtLib ; 
class GNodeLib ; 
class GMergedMesh ; 

class GGEO_API GGeoBase {
    public:
        virtual GScintillatorLib* getScintillatorLib() = 0 ; 
        virtual GSourceLib*       getSourceLib() = 0 ; 
        virtual GBndLib*          getBndLib() = 0 ; 
        virtual GPmtLib*          getPmtLib() = 0 ; 

        virtual const char*       getIdentifier() = 0 ; 
        virtual GMergedMesh*      getMergedMesh(unsigned index) = 0 ; 
        virtual GGeoLib*          getGeoLib() = 0 ;        // GMergedMesh 
        virtual GNodeLib*         getNodeLib() = 0 ;       // GNode/GSolid pv,lv names
};
