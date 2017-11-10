#pragma once

#include <vector>
#include <string>

class Opticks ; 

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

AIMING TO REPLACE THIS WITH CSurfaceLib : SEE :doc:`notes/issues/surface_review`


CSurLib is a constituent of CGeometry that is instanciated with CGeometry. 
The `convert(CDetector* detector)` method is invoked from CGeometry::init 
this creates G4LogicalBorderSurface and G4LogicalSkinSurface together with 
their associated optical surfaces. 
The CDetector argument is used to lookup actual PhysicalVolumes from pv indices 
and actual LogicalVolumes from lv names.

Hmm how to apply to CTestDetector ? PV indices are all different.

* see :doc:`notes/issues/surface_review`


* Tested with CGeometryTest 

**/

class CFG4_API CSurLib 
{
         friend class CDetector ;
         friend class CGeometry ;
    public:
         CSurLib(GSurLib* surlib);
         std::string brief();
         unsigned getNumSur();
         GSur* getSur(unsigned index);
    protected:
         void convert(CDetector* detector);
    private:
         // lookup G4 pv1,pv2 from volume pair indices
         G4LogicalBorderSurface* makeBorderSurface(GSur* sur, unsigned ivp, G4OpticalSurface* os);
         // lookup G4 lv via name 
         G4LogicalSkinSurface*   makeSkinSurface(  GSur* sur, unsigned ilv, G4OpticalSurface* os);
         G4OpticalSurface*       makeOpticalSurface(GSur* sur);
         void addProperties(G4MaterialPropertiesTable* mpt_, GPropertyMap<float>* pmap);
         void setDetector(CDetector* detector);
    private:
         GSurLib*       m_surlib ; 
         Opticks*       m_ok ; 
         bool           m_dbgsurf ; 
         GSurfaceLib*   m_surfacelib ; 
         CDetector*     m_detector ; 

         std::vector<G4LogicalBorderSurface*> m_border ; 
         std::vector<G4LogicalSkinSurface*>   m_skin ; 


};
