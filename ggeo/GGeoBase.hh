#pragma once

#include "GGEO_API_EXPORT.hh"

class GSourceLib ; 
class GScintillatorLib ; 
class GSurfaceLib ; 
class GMaterialLib ; 
class GBndLib ; 
class GGeoLib ; 
class GPmtLib ; 
class GNodeLib ; 
class GMergedMesh ; 

class GGEO_API GGeoBase {
    public:
        virtual GScintillatorLib* getScintillatorLib() const = 0 ; 
        virtual GSourceLib*       getSourceLib() const = 0 ; 
        virtual GSurfaceLib*      getSurfaceLib() const = 0 ; 
        virtual GMaterialLib*     getMaterialLib() const = 0 ; 

        virtual GBndLib*          getBndLib() const = 0 ; 
        virtual GPmtLib*          getPmtLib() const = 0 ; 
        virtual GGeoLib*          getGeoLib() const = 0 ;        // GMergedMesh 
        virtual GNodeLib*         getNodeLib() const = 0 ;       // GNode/GVolume pv,lv names

        virtual const char*       getIdentifier() const = 0 ; 
        virtual GMergedMesh*      getMergedMesh(unsigned index) const = 0 ; 

};
