#pragma once

#include <cstddef>

class Opticks ; 

class GGeoTestConfig ; 
class GGeo ; 
class GGeoLib ; 
class GBndLib ; 
class GMaker ; 
class GMergedMesh ; 

// creates simple test geometries from a commandline specification

class GGeoTest {
    public:
       GGeoTest(Opticks* opticks, GGeoTestConfig* config, GGeo* ggeo=NULL);
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
       Opticks*         m_opticks ; 
       GGeoTestConfig*  m_config ; 
       GGeo*            m_ggeo ; 
       GGeoLib*         m_geolib ; 
       GBndLib*         m_bndlib ; 
       GMaker*          m_maker ; 
       unsigned int     m_verbosity ;

};

inline GGeoTest::GGeoTest(Opticks* opticks, GGeoTestConfig* config, GGeo* ggeo) 
    : 
    m_opticks(opticks),
    m_config(config),
    m_ggeo(ggeo),
    m_geolib(NULL),
    m_bndlib(NULL),
    m_maker(NULL),
    m_verbosity(0)
{
    init();
}



