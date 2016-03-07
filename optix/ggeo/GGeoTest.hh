#pragma once

#include <cstddef>

class GCache ; 
class GGeoTestConfig ; 
class GGeo ; 
class GGeoLib ; 
class GBndLib ; 
class GMaker ; 
class GMergedMesh ; 

class GGeoTest {
    public:
       GGeoTest(GCache* cache, GGeoTestConfig* config);
       void dump(const char* msg="GGeoTest::dump");
       void modifyGeometry();
    private:
       void init();
    private:
       GMergedMesh* create();
    private:
       GMergedMesh* createPmtInBox();
       GMergedMesh* createBoxInBox();
       GMergedMesh* loadPmt();
    private:
       GCache*          m_cache ; 
       GGeoTestConfig*  m_config ; 
       GGeo*            m_ggeo ; 
       GGeoLib*         m_geolib ; 
       GBndLib*         m_bndlib ; 
       GMaker*          m_maker ; 
       unsigned int     m_verbosity ;

};

inline GGeoTest::GGeoTest(GCache* cache, GGeoTestConfig* config) 
    : 
    m_cache(cache),
    m_config(config),
    m_ggeo(NULL),
    m_geolib(NULL),
    m_bndlib(NULL),
    m_maker(NULL),
    m_verbosity(0)
{
    init();
}



