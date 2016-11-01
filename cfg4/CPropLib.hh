#pragma once

#include <cstddef>
#include <string>
#include <map>

class Opticks ;    // okc-
class OpticksHub ; // okg-

// ggeo-
class GBndLib ;
class GMaterialLib ;
class GSurfaceLib ;
class GScintillatorLib ;
class GCSG ; 
class GMaterial ;

template <typename T> class GProperty ; 
template <typename T> class GPropertyMap ; 
template <typename T> class GDomain ; 

// npy-
struct NSlice ; 

// g4-
class G4Material ; 
class G4MaterialPropertiesTable ; 
class G4VPhysicalVolume ;
class G4LogicalBorderSurface ;
class G4OpticalSurface ;
class G4PhysicsVector ;

/**
CPropLib
==========

CPropLib is base class of CMaterialLib which is a constituent of CDetector (eg CTestDector and CGDMLDetector)
that converts GGeo (ie Opticks G4DAE) materials and surfaces into G4 materials and surfaces.

TODO:
------

* remove duplications between CPropLib and tests/CInterpolationTest 


**/


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"
class CFG4_API CPropLib {
   public:
       static const char* SENSOR_MATERIAL ;
   public:
       CPropLib(OpticksHub* hub, int verbosity=0);
   private:
       void init();
       void initCheckConstants(); 
       void initSetupOverrides(); 
   public:
       GSurfaceLib* getSurfaceLib();
   public:
       // GGeo material access
       unsigned int getNumMaterials();
       const GMaterial* getMaterial(unsigned int index);
       const GMaterial* getMaterial(const char* shortname);
       bool hasMaterial(const char* shortname); 
   public:
       std::string getMaterialKeys(const G4Material* mat);
   public:
       GCSG*       getPmtCSG(NSlice* slice);
   public:
       void setGroupvelKludge(bool gk=true);
   public:
       G4LogicalBorderSurface* makeConstantSurface(const char* name, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2, float effi=0.f, float refl=0.f);
       G4LogicalBorderSurface* makeCathodeSurface(const char* name, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2);
   private:
       G4OpticalSurface* makeOpticalSurface(const char* name);

   public:
       // used by CGDMLDetector::addMPT TODO: privatize
       G4MaterialPropertiesTable* makeMaterialPropertiesTable(const GMaterial* kmat);

   protected:
       void addProperties(G4MaterialPropertiesTable* mpt, GPropertyMap<float>* pmap, const char* _keys, bool keylocal=true, bool constant=false);
       void addProperty(G4MaterialPropertiesTable* mpt, const char* matname, const char* lkey,  GProperty<float>* prop );
       void addConstProperty(G4MaterialPropertiesTable* mpt, const char* matname, const char* lkey,  GProperty<float>* prop );
       GProperty<float>* convertVector(G4PhysicsVector* pvec);
       GPropertyMap<float>* convertTable(G4MaterialPropertiesTable* mpt, const char* name);

   protected:
       OpticksHub*        m_hub ; 
       Opticks*           m_ok ; 
       int                m_verbosity ; 

       GBndLib*           m_bndlib ; 
       GMaterialLib*      m_mlib ; 
       GSurfaceLib*       m_slib ; 
       GScintillatorLib*  m_sclib ; 
       GDomain<float>*    m_domain ; 
       float              m_dscale ;  
       GPropertyMap<float>* m_sensor_surface ; 
   protected:
       bool              m_groupvel_kludge ; 
   private:
       std::map<std::string, std::map<std::string, float> > m_const_override ; 

};
#include "CFG4_TAIL.hh"


