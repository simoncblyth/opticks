#pragma once

#include <vector>
#include <string>

template <typename T> class GPropertyMap ; 
class GSur ; 
class GSurLib ; 
class GSurfaceLib ; 

class G4OpticalSurface ; 
class G4LogicalBorderSurface ; 
class G4LogicalSkinSurface ; 
class G4MaterialPropertiesTable ;
class GOpticalSurface ; 

class CDetector ; 

#include "CFG4_API_EXPORT.hh"

/**
CSurLib
========

* Tested with CGeometryTest 

**/

class CFG4_API CSurLib 
{
         friend class CGeometry ;
    public:
         CSurLib(GSurLib* surlib);
         std::string brief();
    protected:
         void convert(CDetector* detector);
         G4LogicalBorderSurface* makeBorderSurface(GSur* sur, unsigned ivp, G4OpticalSurface* os);
         G4LogicalSkinSurface*   makeSkinSurface(  GSur* sur, unsigned ilv, G4OpticalSurface* os);
         G4OpticalSurface*       makeOpticalSurface(GSur* sur);
    private:
         void addProperties(G4MaterialPropertiesTable* mpt_, GPropertyMap<float>* pmap);
         void setDetector(CDetector* detector);
    private:
         GSurLib*       m_surlib ; 
         GSurfaceLib*   m_surfacelib ; 
         CDetector*     m_detector ; 

         std::vector<G4LogicalBorderSurface*> m_border ; 
         std::vector<G4LogicalSkinSurface*>   m_skin ; 


};
