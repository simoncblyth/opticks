#pragma once


class Opticks ; 

class GGeoTestConfig ; 
class GGeo ; 
class GGeoLib ; 
class GBndLib ; 
class GMaker ; 
class GMergedMesh ; 


/**

GGeoTest
=========

Creates simple test geometries from a commandline specification.

**/


#include "GGEO_API_EXPORT.hh"
class GGEO_API GGeoTest {
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


