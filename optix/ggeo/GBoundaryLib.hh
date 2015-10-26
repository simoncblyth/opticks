#pragma once

#include <map>
#include <vector>
#include <string>

#include "GVector.hh"
#include "GDomain.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"

class GBoundary ; 
class GBuffer ; 
class GBoundaryLibMetadata ; 
class GBuffer ; 
class GItemIndex ; 
//class GMaterialIndex ; 
//class GSurfaceIndex ; 

#include "GPropertyLib.hh"


class GBoundaryLib : public GPropertyLib {
  public:
     typedef std::map<unsigned int, std::string> Index_t ;
     enum {
         optical_index, 
         optical_type, 
         optical_finish, 
         optical_value }; 

  public:
    static const char* wavelength ; 
    static const char* reemission ; 
    static const char* optical ; 
  public:
    // standard property prefixes
    static const char* inner; 
    static const char* outer; 
  public:
    static unsigned int NUM_QUAD ; 

    static float        SURFACE_UNSET ; 
    static float        EXTRA_UNSET ; 
  public:
    // 4 standard material property names : interleaved into float4 wavelength texture
    static const char* refractive_index ; 
    static const char* absorption_length ; 
    static const char* scattering_length ; 
    static const char* reemission_prob ; 
  public:
    // 4 standard surface property names : interleaved into float4 wavelength texture
    static const char* detect ;
    static const char* absorb ;
    static const char* reflect_specular ;
    static const char* reflect_diffuse ;
  public:
     // extra
    static const char* reemission_cdf ; 
    static const char* extra_x ; 
    static const char* extra_y ; 
    static const char* extra_z ; 
    static const char* extra_w ; 
  public:
    // some model-mismatch translation required for surface properties
    static const char* EFFICIENCY ; 
    static const char* REFLECTIVITY ; 
  public:
     // workings for extra
    static const char* slow_component; 
    static const char* fast_component; 

  public:
    static const char* keymap ;

  private:
    // remission handling : needs energywise 1/wavelength[::-1] domain for CDF 
    static const char* scintillators ;
    static std::vector<std::string>* vscintillators ;
    static const char* reemissionkey ;
    static std::vector<std::string>* vreemissionkey;

  public:
      GBoundaryLib(GCache* cache);
      virtual ~GBoundaryLib();
  private:
      void init();
  public:
      // primary methods : lifecycle
      // void setStandardDomain(GDomain<float>* standard_domain);
      GBoundary* getOrCreate(
                      GPropertyMap<float>* imaterial, 
                      GPropertyMap<float>* omaterial, 
                      GPropertyMap<float>* isurface, 
                      GPropertyMap<float>* osurface,
                      GPropertyMap<float>* iextra, 
                      GPropertyMap<float>* oextra
                 );
      static unsigned int getLine(unsigned int isub, unsigned int ioff);
      unsigned int getLineMin();
      unsigned int getLineMax();
  
      void          countMaterials();

      // canonically invoked during the flattening by GMergedMesh::create
      void          createWavelengthAndOpticalBuffers();  
      GBuffer*      createReemissionBuffer(GPropertyMap<float>* scint);  

  public:
      // methods for debug
      void setFakeEfficiency(float fake_efficiency);

  public:
      // methods supporting save/load from file
      static GBoundaryLib* load(GCache* cache);  
      void save(const char* dir);

  private:
      static void nameConstituents(std::vector<std::string>& names);
      void loadBuffers(const char* dir);
      void saveBuffer(const char* path, const char* name, GBuffer* buffer);
      void loadBuffer(const char* path, const char* name);
      bool isIntBuffer(const char* name);
      bool isUIntBuffer(const char* name);
      bool isFloatBuffer(const char* name);
      void setBuffer(const char* name, GBuffer* buffer);
      GBuffer* getBuffer(const char* name);
  public:
      void         setWavelengthBuffer(GBuffer* buffer);
      void         setReemissionBuffer(GBuffer* buffer);
      void         setOpticalBuffer(GBuffer* buffer);
      GBuffer*     getWavelengthBuffer();
      GBuffer*     getReemissionBuffer();
      GBuffer*     getOpticalBuffer();
  public:
      void         setColorBuffer(GBuffer* buffer);
      GBuffer*     getColorBuffer();
      void         setColorDomain(guint4 domain);
      guint4       getColorDomain();
  public:
      // reemission handling 
      bool isScintillator(std::string& matShortName);
      bool isReemissionKey(std::string& lkey);
      GProperty<float>* constructReemissionCDF(GPropertyMap<float>* pmap);
      GProperty<float>* constructInvertedReemissionCDF(GPropertyMap<float>* pmap);
  public:
      // primary methods : querying 

      unsigned int getNumBoundary();
      GBoundary* getBoundary(unsigned int index); 
      void dumpSurfaces(const char* msg="GBoundaryLib::dumpSurfaces");
      void Summary(const char* msg="GBoundaryLib::Summary");
      const char* getDigest(unsigned int index);
  public:
      GBoundaryLibMetadata* getMetadata(); // populated by createWavelengthBuffer
      GItemIndex*           getMaterials(); 
      GItemIndex*           getSurfaces(); 

  public:
      std::vector<std::string> splitString(std::string keys);

  private:
      // used for by "get" for standardization of boundaries, ready for serializing into wavelengthBuffer
      GBoundary* createStandardBoundary(GBoundary* boundary);
      void standardizeMaterialProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix);
      void standardizeSurfaceProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix);
      void standardizeExtraProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix);
      bool checkSurface( GPropertyMap<float>* surf, const char* prefix );
      void dumpSurface( GPropertyMap<float>* surf, const char* prefix, const char* msg="GBoundaryLib::dumpSurface");

  private:
      // support for standardization 
      void defineDefaults(GPropertyMap<float>* defaults);
      void import();


      GProperty<float>* getRamp();

      char*  digest(std::vector<GProperty<float>*>& props);
      std::string digestString(std::vector<GProperty<float>*>& props);

      void digestDebug(GBoundary* boundary, unsigned int isub);


  public:
      void setMetadata(GBoundaryLibMetadata* meta); 

  public:
      void          dumpWavelengthBuffer(int wline=-1);
      static void   dumpWavelengthBuffer(int wline, GBuffer* buffer, GBoundaryLibMetadata* metadata, unsigned int numBoundary, unsigned int domainLength);
  public:
      void          dumpOpticalBuffer(int wline=-1);
      static void   dumpOpticalBuffer(   int wline, GBuffer* buffer, GBoundaryLibMetadata* metadata, unsigned int numBoundary);

  public:
      void addToIndex(GPropertyMap<float>* obj);
      void dumpIndex(const char* msg="GBoundaryLib::dumpIndex");
      void saveIndex(const char* cachedir, const char* filename="GBoundaryLibIndex.json");

  public:
      GPropertyMap<float>* createStandardProperties(const char* name, GBoundary* boundary);
      void checkMaterialProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* prefix);
      void checkSurfaceProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* prefix);
      void checkExtraProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* prefix);

  public:
      static unsigned int getNumProp();
      static unsigned int getNumQuad();
      const char* surfacePropertyName(unsigned int i);
      const char* extraPropertyName(unsigned int i);
      char* propertyName(unsigned int p, unsigned int i);
      std::string propertyNameString(unsigned int p, unsigned int i);

  public:
      GBoundary* loadBoundary(float* subData, unsigned int isub);


  private:
      std::map<std::string, GBoundary*>    m_registry ; 
      std::vector<std::string>             m_keys ; 
      Index_t                              m_index ; 

      bool                   m_standard ;     // transitional : keeping this set to true
      unsigned int           m_num_quad ; 
      GProperty<float>*      m_ramp ;  
      GBoundaryLibMetadata*  m_meta ;
      GItemIndex*            m_materials ;
      GItemIndex*            m_surfaces ;
  private:
      std::vector<std::string> m_names ;          // names of constituent persistable buffers 
      GBuffer*               m_wavelength_buffer ;
      GBuffer*               m_reemission_buffer ;
      GBuffer*               m_optical_buffer ;
  private:
      // transients
      GBuffer*               m_color_buffer ;
      guint4                 m_color_domain ; 
      float                  m_fake_efficiency ; 

};


inline GBoundaryLib::GBoundaryLib(GCache* cache) 
          : 
          GPropertyLib(cache, "GBoundaryLib"),
          m_standard(true), 
          m_num_quad(6), 
          m_ramp(NULL), 
          m_meta(NULL), 
          m_materials(NULL), 
          m_surfaces(NULL), 
          m_wavelength_buffer(NULL),
          m_reemission_buffer(NULL),
          m_optical_buffer(NULL),
          m_color_buffer(NULL),
          m_color_domain(0,0,0,0),
          m_fake_efficiency(-1.f)
{
     init();
}

inline GBoundaryLib::~GBoundaryLib()
{
}


inline GBoundaryLibMetadata* GBoundaryLib::getMetadata()
{
    return m_meta ; 
}

inline void GBoundaryLib::setMetadata(GBoundaryLibMetadata* meta)
{
    m_meta = meta ; 
}

inline GItemIndex* GBoundaryLib::getMaterials()
{
    return m_materials ; 
}
inline GItemIndex* GBoundaryLib::getSurfaces()
{
    return m_surfaces ; 
}





inline GBuffer* GBoundaryLib::getWavelengthBuffer()
{
    return m_wavelength_buffer ; 
}

inline GBuffer* GBoundaryLib::getOpticalBuffer()
{
    return m_optical_buffer ; 
}
inline void GBoundaryLib::setOpticalBuffer(GBuffer* optical_buffer)
{
    m_optical_buffer = optical_buffer ; 
}

inline GBuffer* GBoundaryLib::getReemissionBuffer()
{
    return m_reemission_buffer ; 
}
inline void GBoundaryLib::setReemissionBuffer(GBuffer* reemission_buffer)
{
    m_reemission_buffer = reemission_buffer ; 
}


inline GBuffer* GBoundaryLib::getColorBuffer()
{
    return m_color_buffer ; 
}
inline void GBoundaryLib::setColorBuffer(GBuffer* color_buffer)
{
    m_color_buffer = color_buffer ; 
}


inline guint4 GBoundaryLib::getColorDomain()
{
    return m_color_domain  ; 
}
inline void GBoundaryLib::setColorDomain(guint4 color_domain)
{
    m_color_domain = color_domain ; 
}





inline unsigned int GBoundaryLib::getNumProp()
{
    return NUM_QUAD*4 ; 
}
inline unsigned int GBoundaryLib::getNumQuad()
{
    return NUM_QUAD ; 
}


inline GProperty<float>* GBoundaryLib::getRamp()
{
   return m_ramp ;
}


inline void GBoundaryLib::setFakeEfficiency(float fake_efficiency)
{
    m_fake_efficiency = fake_efficiency ; 
}

