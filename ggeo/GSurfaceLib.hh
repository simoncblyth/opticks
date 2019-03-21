#pragma once

#include <vector>

/** 
GSurfaceLib
==============

Skin and Border surfaces have an associated optical surface 
that is lodged inside GPropertyMap
in addition to 1(for skin) or 2(for border) volume names

* huh : where are these names persisted ?
    

ISSUE
-------

* domain not persisted, so have to just assume that are using 
  standard one at set on load ?



**/

class NMeta ; 
struct guint4 ; 
class GOpticalSurface ; 
class GSkinSurface ; 
class GBorderSurface ; 
class GItemList ; 
template<typename T> class GProperty ; 

#include "GPropertyLib.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"
#include "plog/Severity.h"


class GGEO_API GSurfaceLib : public GPropertyLib {
   public:
       static const plog::Severity LEVEL ; 
       static const char* propertyName(unsigned int k);
       // 4 standard surface property names : interleaved into float4 wavelength texture
  public:
       static const char* detect ;
       static const char* absorb ;
       static const char* reflect_specular ;
       static const char* reflect_diffuse ;
  public:
       static const char* extra_x ; 
       static const char* extra_y ; 
       static const char* extra_z ; 
       static const char* extra_w ; 
  public:
       static const char* AssignSurfaceType( NMeta* surfmeta );
       static const char* BORDERSURFACE ;  
       static const char* SKINSURFACE ;  
       static const char* TESTSURFACE ;  
  public:
       static const char* BPV1 ;  
       static const char* BPV2 ;  
       static const char* SSLV ;  
  public:
       // some model-mismatch translation required for surface properties
       static const char* EFFICIENCY ; 
       static const char* REFLECTIVITY ; 
   public:
       static bool NameEndsWithSensorSurface(const char* name);
       static const char* NameWithoutSensorSurface(const char* name);

       static const char*  SENSOR_SURFACE ;
       static float        SURFACE_UNSET ; 
       static const char* keyspec ;
   public:
       void save();
       static GSurfaceLib* load(Opticks* ok);
   public:
       GSurfaceLib(Opticks* ok, GSurfaceLib* basis=NULL); 
       GSurfaceLib(GSurfaceLib* other, GDomain<float>* domain=NULL, GSurfaceLib* basis=NULL );  // interpolating copy ctor
   private:
       void init();
       void initInterpolatingCopy(GSurfaceLib* src, GDomain<float>* domain);
   public:
       GSurfaceLib* getBasis() const ;
       void         setBasis(GSurfaceLib* basis);
   public:
       void Summary(const char* msg="GSurfaceLib::Summary");
       void dump(const char* msg="GSurfaceLib::dump");
       void dump(GPropertyMap<float>* surf);
       void dump(GPropertyMap<float>* surf, const char* msg);
       void dump(unsigned int index);
       std::string desc() const ; 
   public:
       // concretization of GPropertyLib
       void defineDefaults(GPropertyMap<float>* defaults); 
       NPY<float>* createBuffer();
       NMeta*      createMeta();
       GItemList*  createNames();
   public:
       NPY<float>* createBufferForTex2d();
       NPY<float>* createBufferOld();
   public:
      // methods for debug
       void setFakeEfficiency(float fake_efficiency);
       GPropertyMap<float>* makePerfect(const char* name, float detect_, float absorb_, float reflect_specular_, float reflect_diffuse_);
       void addPerfectSurfaces();
    public:
        // methods to assist with de-conflation of surface props and location
        void addBorderSurface(GPropertyMap<float>* surf, const char* pv1, const char* pv2, bool direct );
        void addSkinSurface(GPropertyMap<float>* surf, const char* sslv_, bool direct );
    public:
        // used from GGeoTest 
        GPropertyMap<float>* getBasisSurface(const char* name) const ; 
        void relocateBasisBorderSurface(const char* name, const char* bpv1, const char* bpv2);
        void relocateBasisSkinSurface(const char* name, const char* sslv);
    public:
        void add(GSkinSurface* ss);
        void add(GBorderSurface* bs);
        void add(GPropertyMap<float>* surf);
   private:
       void addDirect(GPropertyMap<float>* surf);
   public:
       void sort();
       bool operator()(const GPropertyMap<float>* a_, const GPropertyMap<float>* b_);
   public:
       guint4               getOpticalSurface(unsigned int index);  // zero based index
       GPropertyMap<float>* getSensorSurface(unsigned int offset=0);  // 0: first, 1:second 
   public:
       // Check for a surface of specified name of index in m_surfaces vector
       // NB: changed behaviour, formerly named access only worked after closing
       // the lib as used the names buffer     
       GPropertyMap<float>* getSurface(unsigned int index) const ;         // zero based index
       GPropertyMap<float>* getSurface(const char* name) const ;        
       bool hasSurface(unsigned int index) const ; 
       bool hasSurface(const char* name) const ; 
       GProperty<float>* getSurfaceProperty(const char* name, const char* prop) const ;
   private:
       guint4               createOpticalSurface(GPropertyMap<float>* src);
       GPropertyMap<float>* createStandardSurface(GPropertyMap<float>* src);
       bool checkSurface( GPropertyMap<float>* surf);
   public:
      // unlike former GBoundaryLib optical buffer one this is surface only
       NPY<unsigned int>* createOpticalBuffer();  
       void importOpticalBuffer(NPY<unsigned int>* ibuf);
       void saveOpticalBuffer();
       void loadOpticalBuffer();
       void setOpticalBuffer(NPY<unsigned int>* ibuf);
       NPY<unsigned int>* getOpticalBuffer();
   public:
       unsigned getNumSurfaces() const ;
       bool isSensorSurface(unsigned int surface); // name suffix based, see AssimpGGeo::convertSensor
   public:
       void import();
   private:
       void dumpMeta(const char* msg="GSurfaceLib::dumpMeta") const ;
       void importOld();
       void importForTex2d();
       void import( GPropertyMap<float>* surf, float* data, unsigned int nj, unsigned int nk, unsigned int jcat=0 );

   public:
       // simple collections relocated from GGeo
       unsigned getNumBorderSurfaces() const ;
       unsigned getNumSkinSurfaces() const ;
       GSkinSurface* getSkinSurface(unsigned index) const ;
       GBorderSurface* getBorderSurface(unsigned index) const ;

       void addRaw(GBorderSurface* surface); 
       void addRaw(GSkinSurface* surface);
       unsigned getNumRawBorderSurfaces() const ;
       unsigned getNumRawSkinSurfaces() const ;

       GSkinSurface* findSkinSurface(const char* lv) const ;
       void dumpSkinSurface(const char* msg="GSurfaceLib::dumpSkinSurface") const ;

       GBorderSurface* findBorderSurface(const char* pv1, const char* pv2) const ;
       void dumpRawSkinSurface(const char* name) const ;
       void dumpRawBorderSurface(const char* name) const ;

   private:
       std::vector<GPropertyMap<float>*>       m_surfaces ; 
       float                                   m_fake_efficiency ; 
       NPY<unsigned int>*                      m_optical_buffer ; 
       GSurfaceLib*                            m_basis ; 
       bool                                    m_dbgsurf ; 
       plog::Severity                          m_level ; 

   private:
       // relocated from GGeo
       std::vector<GSkinSurface*>    m_skin_surfaces ; 
       std::vector<GSkinSurface*>    m_sensor_skin_surfaces ; 
       std::vector<GBorderSurface*>  m_border_surfaces ; 
        // _raw mainly for debug
       std::vector<GSkinSurface*>    m_skin_surfaces_raw ; 
       std::vector<GBorderSurface*>  m_border_surfaces_raw ; 


};

#include "GGEO_TAIL.hh"


