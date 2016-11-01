#pragma once

#include <map>
#include <vector>
#include <string>

class Opticks ; 

struct guint4 ; 
template <typename T> class GPropertyMap ;
class GMaterialLib ; 
class GSurfaceLib ; 

//
// *GBndLib* differs from *GMaterialLib* and *GSurfaceLib* in that 
// creation of its float buffer needs to be deferred post cache 
// in order to allow dynamic addition of boundaries for eg analytic
// geometry inside-outs and for test boxes 
//
// Instead the index buffer is used for persisting, which contains
// indices of materials and surfaces imat/omat/isur/osur 
//
// The boundary buffer is created dynamically by pulling the 
// relevant bytes from the material and surface libs. 
// 
// Former *GBoundaryLib* encompassed uint4 optical_buffer that 
// contained surface properties from GOpticalSurface, that
// is now moved to *GSurfaceLib*   
//

#include "GPropertyLib.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GBndLib : public GPropertyLib {
  public:
       enum {
               OMAT,
               OSUR,
               ISUR,
               IMAT
            };
  public:
       void save();
       static GBndLib* load(Opticks* cache, bool constituents=false);
  public:
       GBndLib(Opticks* cache);
  private:
       void init(); 
  public:
       unsigned int getNumBnd();
  public:
       std::string description(const guint4& bnd);
       std::string shortname(const guint4& bnd);
       std::string shortname(unsigned int boundary);
       bool contains(const guint4& bnd);
       unsigned int index(const guint4& bnd);
  public:
       // boundary index lookups
       guint4 getBnd(unsigned int boundary);
  public:
       unsigned getOuterMaterial(unsigned boundary);
       unsigned getOuterSurface(unsigned boundary);
       unsigned getInnerSurface(unsigned boundary);
       unsigned getInnerMaterial(unsigned boundary);
  public:
       const char* getOuterMaterialName(unsigned int boundary);
       const char* getOuterSurfaceName(unsigned int boundary);
       const char* getInnerSurfaceName(unsigned int boundary);
       const char* getInnerMaterialName(unsigned int boundary);
  public:
       const char* getOuterMaterialName(const char* spec);
       const char* getOuterSurfaceName(const char* spec);
       const char* getInnerSurfaceName(const char* spec);
       const char* getInnerMaterialName(const char* spec);
  public:
       guint4 parse( const char* spec, bool flip=false);
       bool contains( const char* spec, bool flip=false);
  public:
       unsigned int addBoundary( const char* spec, bool flip=false) ;
       unsigned int addBoundary( const char* omat, const char* osur, const char* isur, const char* imat) ;
  private:
       // Bnd are only added if not already present
       void add(const guint4& bnd);
       guint4 add(const char* spec, bool flip=false);
       guint4 add(const char* omat, const char* osur, const char* isur, const char* imat);
       guint4 add(unsigned int omat, unsigned int osur, unsigned int isur, unsigned int imat);
  public:
       void loadIndexBuffer();
       void importIndexBuffer();
  public:
       void saveIndexBuffer();
       void saveOpticalBuffer();
  public:
       NPY<unsigned int>* createIndexBuffer();
       NPY<unsigned int>* createOpticalBuffer();
  public:
       bool hasIndexBuffer();
       NPY<unsigned int>* getIndexBuffer();
       NPY<unsigned int>* getOpticalBuffer();
  public:
       void setIndexBuffer(NPY<unsigned int>* index_buffer);
       void setOpticalBuffer(NPY<unsigned int>* optical_buffer);
  public:
       const std::map<std::string, unsigned int>& getMaterialLineMap();
       void dumpMaterialLineMap(const char* msg="GBndLib::dumpMaterialLineMap"); 
  private:
       void fillMaterialLineMap(std::map<std::string, unsigned int>& msu);
       void dumpMaterialLineMap(std::map<std::string, unsigned int>& msu, const char* msg="GBndLib::dumpMaterialLineMap");
  public:
       unsigned int getMaterialLine(const char* shortname);
       static unsigned int getLine(unsigned int ibnd, unsigned int iquad);
       unsigned int getLineMin();
       unsigned int getLineMax();
  public:
       void createDynamicBuffers();
   public:
       NPY<float>* createBufferForTex2d();
       NPY<float>* createBufferOld();
  public:
       GItemList* createNames();
       NPY<float>* createBuffer();
       void import();
       void sort();
       void defineDefaults(GPropertyMap<float>* defaults);
  public:
       void Summary(const char* msg="GBndLib::Summary");
       void dump(const char* msg="GBndLib::dump");
       void dumpBoundaries(std::vector<unsigned int>& boundaries, const char* msg="GBndLib::dumpBoundaries");
  public:
       void setMaterialLib(GMaterialLib* mlib);
       void setSurfaceLib(GSurfaceLib* slib);
       GMaterialLib* getMaterialLib();
       GSurfaceLib*  getSurfaceLib();
  private:
       GMaterialLib*        m_mlib ; 
       GSurfaceLib*         m_slib ; 
       std::vector<guint4>  m_bnd ; 
       NPY<unsigned int>*   m_index_buffer ;  
       NPY<unsigned int>*   m_optical_buffer ;  
       std::map<std::string, unsigned int> m_materialLineMap ;

};

#include "GGEO_TAIL.hh"


