#pragma once

#include <vector>
#include <string>

class Opticks ; 

template <typename T> class GPropertyMap ; 
class GSurfaceLib ; 

class G4OpticalSurface ; 
class G4LogicalBorderSurface ; 
class G4LogicalSkinSurface ; 
class G4MaterialPropertiesTable ;
class GOpticalSurface ; 

class CDetector ; 

#include "plog/Severity.h"
#include "CFG4_API_EXPORT.hh"

/**
CSurfaceLib
=============

See :doc:`notes/issues/surface_review` 

* CSurfaceLib is aiming to eliminate the kludgy classes (GSur/GSurLib/CSurLib)
  by using a simpler direct from GSurfaceLib approach : made possible
  by improved GPropLib NMeta persisting 

Predecessor CSurLib
~~~~~~~~~~~~~~~~~~~~~

Canonical m_csurlib was a constituent of CDetector that is instanciated  
The `convert(CDetector* detector)` method is invoked from CGeometry::init 
this creates G4LogicalBorderSurface and G4LogicalSkinSurface together with 
their associated optical surfaces. 
The CDetector argument is used to lookup actual PhysicalVolumes from pv indices 
and actual LogicalVolumes from lv names.

Hmm how to apply to CTestDetector ? PV indices are all different.

* see :doc:`notes/issues/surface_review`


* Tested with CGeometryTest 

**/

class CFG4_API CSurfaceLib 
{
         friend class CDetector ;
         friend class CGeometry ;
    public:
         CSurfaceLib(GSurfaceLib* surlib);
         std::string brief();
    protected:
         void convert(CDetector* detector, bool exclude_sensors);
    private:
         G4OpticalSurface*       makeOpticalSurface(GPropertyMap<float>* surf);
         G4LogicalBorderSurface* makeBorderSurface( GPropertyMap<float>* surf, G4OpticalSurface* os);
         G4LogicalSkinSurface*   makeSkinSurface(   GPropertyMap<float>* surf, G4OpticalSurface* os);

         void addProperties(G4MaterialPropertiesTable* mpt_, GPropertyMap<float>* pmap);
         void setDetector(CDetector* detector);
    private:
         GSurfaceLib*   m_surfacelib ; 
         Opticks*       m_ok ; 
         bool           m_dbgsurf ; 
         CDetector*     m_detector ; 
         plog::Severity m_level ; 

         std::vector<G4LogicalBorderSurface*> m_border ; 
         std::vector<G4LogicalSkinSurface*>   m_skin ; 
};

