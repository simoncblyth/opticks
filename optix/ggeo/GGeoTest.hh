#pragma once

#include <cstddef>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <map>

struct NSlice ; 

class GCache ; 
class GGeo ; 
class GGeoLib ; 
class GBndLib ; 
class GMergedMesh ; 

class GGeoTest {
    public:
       typedef enum { 
                      MODE, 
                      FRAME, 
                      DIMENSIONS, 
                      BOUNDARY, 
                      SHAPE, 
                      SLICE, 
                      ANALYTIC, 
                      UNRECOGNIZED } Param_t ;

       typedef std::pair<std::string,std::string> KV ; 
       static const char* DEFAULT_CONFIG ; 
    public:
       static const char* MODE_; 
       static const char* FRAME_ ; 
       static const char* DIMENSIONS_; 
       static const char* BOUNDARY_ ; 
       static const char* SHAPE_ ; 
       static const char* SLICE_ ; 
       static const char* ANALYTIC_ ; 
    public:
       GGeoTest(GCache* cache);
       void configure(const char* config=NULL);
       void modifyGeometry();
       void dump(const char* msg="GGeoTest::dump");
       std::vector<std::pair<std::string, std::string> >& getCfg();
    private:
       void init();
       Param_t getParam(const char* k);
       void set(Param_t p, const char* s);
    private:
       void setMode(const char* s);
       void setFrame(const char* s);
       void setDimensions(const char* s);
       void setShape(const char* s);
       void addBoundary(const char* s);
       void setSlice(const char* s);
       void setAnalytic(const char* s);
    private:
       GMergedMesh* createPmtInBox();
       GMergedMesh* createBoxInBox();
    private:
       GCache*      m_cache ; 
       const char*  m_config ; 
       GGeo*        m_ggeo ; 
       GGeoLib*     m_geolib ; 
       GBndLib*     m_bndlib ; 
    private:
       const char*  m_mode ; 
       NSlice*      m_slice ; 
       glm::ivec4   m_frame ;
       glm::vec4    m_dimensions ;
       glm::ivec4   m_analytic ;
       glm::ivec4   m_shape ;
       std::vector<std::string> m_boundaries ; 
       std::vector<KV> m_cfg ; 

};

inline GGeoTest::GGeoTest(GCache* cache) 
    : 
    m_cache(cache),
    m_config(NULL),
    m_ggeo(NULL),
    m_geolib(NULL),
    m_bndlib(NULL),
    m_mode(NULL),
    m_slice(NULL),
    m_frame(0,0,0,0),
    m_analytic(0,0,0,0),
    m_shape('U','U','U','U')
{
    init();
}

inline std::vector<std::pair<std::string, std::string> >& GGeoTest::getCfg()
{
    return m_cfg ; 
}


