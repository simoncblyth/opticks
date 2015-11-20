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
class GMaker ; 
class GMergedMesh ; 

class GGeoTest {
    public:
       typedef enum { 
                      MODE, 
                      FRAME, 
                      BOUNDARY, 
                      PARAMETERS, 
                      SHAPE, 
                      SLICE, 
                      ANALYTIC, 
                      DEBUG,
                      UNRECOGNIZED } Arg_t ;

       typedef std::pair<std::string,std::string> KV ; 
       static const char* DEFAULT_CONFIG ; 
    public:
       static const char* MODE_; 
       static const char* FRAME_ ; 
       static const char* BOUNDARY_ ; 
       static const char* PARAMETERS_ ; 
       static const char* SHAPE_ ; 
       static const char* SLICE_ ; 
       static const char* ANALYTIC_ ; 
       static const char* DEBUG_ ; 
    public:
       GGeoTest(GCache* cache);
       void configure(const char* config=NULL);
       void modifyGeometry();
       void dump(const char* msg="GGeoTest::dump");
       std::vector<std::pair<std::string, std::string> >& getCfg();
    private:
       void init();
       Arg_t getArg(const char* k);
       void set(Arg_t arg, const char* s);
    private:
       void setMode(const char* s);
       void setFrame(const char* s);
       void setShape(const char* s);
       void addBoundary(const char* s);
       void addParameters(const char* s);
       void setSlice(const char* s);
       void setAnalytic(const char* s);
       void setDebug(const char* s);
    private:
       glm::vec4 getParameters(unsigned int i);
    private:
       GMergedMesh* createPmtInBox();
       GMergedMesh* createBoxInBox();
    private:
       GCache*      m_cache ; 
       const char*  m_config ; 
       GGeo*        m_ggeo ; 
       GGeoLib*     m_geolib ; 
       GBndLib*     m_bndlib ; 
       GMaker*      m_maker ; 
    private:
       const char*  m_mode ; 
       NSlice*      m_slice ; 
       glm::ivec4   m_frame ;
       glm::ivec4   m_analytic ;
       glm::ivec4   m_shape ;
       glm::vec4    m_debug ;
       std::vector<std::string> m_boundaries ; 
       std::vector<glm::vec4>   m_parameters ; 
       std::vector<KV> m_cfg ; 

};

inline GGeoTest::GGeoTest(GCache* cache) 
    : 
    m_cache(cache),
    m_config(NULL),
    m_ggeo(NULL),
    m_geolib(NULL),
    m_bndlib(NULL),
    m_maker(NULL),
    m_mode(NULL),
    m_slice(NULL),
    m_frame(0,0,0,0),
    m_analytic(0,0,0,0),
    m_shape('U','U','U','U'),
    m_debug(1.f,0.f,0.f,0.f)
{
    init();
}

inline std::vector<std::pair<std::string, std::string> >& GGeoTest::getCfg()
{
    return m_cfg ; 
}


