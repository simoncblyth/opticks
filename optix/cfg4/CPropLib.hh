#pragma once

#include <cstddef>
#include <string>
#include <map>

// okc-
class Opticks; 


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

// CPropLib is a constituent of CTestDetector that converts
// GGeo materials and surfaces into G4 materials and surfaces
//
//  TODO: this need simplification
//
//     * far too much public API
//     * moving to convert internally all at once approach 
//       and provide simple accessors
//
//     * maybe split off conversion "reconstruction" into separate class
//


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"
class CFG4_API CPropLib {
   public:
       static const char* SENSOR_MATERIAL ;
   public:
       CPropLib(Opticks* opticks, int verbosity=0);
   private:
       void init();
       void checkConstants(); 
       void setupOverrides(); 
       void convert();
   public:
       // GGeo material access
       unsigned int getNumMaterials();
       const GMaterial* getMaterial(unsigned int index);
       const GMaterial* getMaterial(const char* shortname);
       bool hasMaterial(const char* shortname); 
   public:
       // G4 material access
       const G4Material* getG4Material(const char* shortname);
       std::string getMaterialKeys(const G4Material* mat);
   public:
       const G4Material* makeInnerMaterial(const char* spec);
       const G4Material* makeMaterial(const char* matname);
       GCSG*       getPmtCSG(NSlice* slice);
   public:
       unsigned int getMaterialIndex(const G4Material* material);
       const char*  getMaterialName(unsigned int index);
       std::string MaterialSequence(unsigned long long seqmat);
       void setGroupvelKludge(bool gk=true);
   public:
       G4LogicalBorderSurface* makeConstantSurface(const char* name, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2, float effi=0.f, float refl=0.f);
       G4LogicalBorderSurface* makeCathodeSurface(const char* name, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2);
   private:
       G4OpticalSurface* makeOpticalSurface(const char* name);
   private:
       const G4Material* convertMaterial(const GMaterial* kmat);
       G4Material* makeVacuum(const char* name);
       G4Material* makeWater(const char* name);
   public:
       // used by CGDMLDetector::addMPT TODO: privatize
       G4MaterialPropertiesTable* makeMaterialPropertiesTable(const GMaterial* kmat);
   private: 
       void addProperties(G4MaterialPropertiesTable* mpt, GPropertyMap<float>* pmap, const char* _keys, bool keylocal=true, bool constant=false);
       void addProperty(G4MaterialPropertiesTable* mpt, const char* matname, const char* lkey,  GProperty<float>* prop );
       void addConstProperty(G4MaterialPropertiesTable* mpt, const char* matname, const char* lkey,  GProperty<float>* prop );
   public: 
       void dump(const char* msg="CPropLib::dump");
       void dump(const GMaterial* mat, const char* msg="CPropLib::dump");
       void dumpMaterials(const char* msg="CPropLib::dumpMaterials");
       void dumpMaterial(const G4Material* mat, const char* msg="CPropLib::dumpMaterial");
   private:
       GProperty<float>* convertVector(G4PhysicsVector* pvec);
       GPropertyMap<float>* convertTable(G4MaterialPropertiesTable* mpt, const char* name);

   private:
       Opticks*           m_opticks ; 
       int                m_verbosity ; 
       GBndLib*           m_bndlib ; 
       GMaterialLib*      m_mlib ; 
       GSurfaceLib*       m_slib ; 
       GScintillatorLib*  m_sclib ; 
       GDomain<float>*    m_domain ; 
       float              m_dscale ;  

       GPropertyMap<float>* m_sensor_surface ; 

       std::map<const GMaterial*, const G4Material*>   m_ggtog4 ; 
       std::map<const G4Material*, unsigned int> m_g4toix ; 
       std::map<unsigned int, std::string> m_ixtoname ; 

       bool              m_groupvel_kludge ; 
   private:
       std::map<std::string, const G4Material*>   m_g4mat ; 
       std::map<std::string, std::map<std::string, float> > m_const_override ; 

};
#include "CFG4_TAIL.hh"


