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
       GMergedMesh* createPmtInBox();
       GMergedMesh* createBoxInBox();
    private:
       GCache*          m_cache ; 
       GGeoTestConfig*  m_config ; 
       GGeo*            m_ggeo ; 
       GGeoLib*         m_geolib ; 
       GBndLib*         m_bndlib ; 
       GMaker*          m_maker ; 

};

inline GGeoTest::GGeoTest(GCache* cache, GGeoTestConfig* config) 
    : 
    m_cache(cache),
    m_config(config),
    m_ggeo(NULL),
    m_geolib(NULL),
    m_bndlib(NULL),
    m_maker(NULL)
{
    init();
}



