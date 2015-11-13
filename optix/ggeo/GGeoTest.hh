#pragma once

#include <cstddef>
#include <glm/glm.hpp>
#include <string>
#include <vector>

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
                      SLICE, 
                      UNRECOGNIZED } Param_t ;

       static const char* DEFAULT_CONFIG ; 
    public:
       static const char* MODE_; 
       static const char* FRAME_ ; 
       static const char* DIMENSIONS_; 
       static const char* BOUNDARY_ ; 
       static const char* SLICE_ ; 
    public:
       GGeoTest(GCache* cache);
       void configure(const char* config=NULL);
       void modifyGeometry();
       void dump(const char* msg="GGeoTest::dump");
    private:
       void init();
       Param_t getParam(const char* k);
       void set(Param_t p, const char* s);
    private:
       void setMode(const char* s);
       void setFrame(const char* s);
       void setDimensions(const char* s);
       void addBoundary(const char* s);
       void setSlice(const char* s);
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
       std::vector<std::string> m_boundaries ; 
};

inline GGeoTest::GGeoTest(GCache* cache) 
    : 
    m_cache(cache),
    m_config(NULL),
    m_ggeo(NULL),
    m_geolib(NULL),
    m_bndlib(NULL),
    m_mode(NULL),
    m_slice(NULL)
{
    init();
}


