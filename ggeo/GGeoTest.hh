#pragma once

struct NLODConfig ; 
class Opticks ; 

class GGeoTestConfig ; 
class GGeo ; 
class GGeoLib ; 
class GBndLib ; 
class GMaker ; 
class GMergedMesh ; 
class GSolid ; 

/**

GGeoTest
=========

Creates simple test geometries from a commandline specification.

The canonical *GGeo* member *m_geotest* instance of *GGeoTest* is only 
instanciated when the `--test` option is used causing the running 
of `GGeo::modifyGeometry`

Controlled from OpticksGeometry::modifyGeometry



**/


#include "GGEO_API_EXPORT.hh"
class GGEO_API GGeoTest {
    public:
       GGeoTest(Opticks* ok, GGeoTestConfig* config, GGeo* ggeo=NULL);
       void dump(const char* msg="GGeoTest::dump");
       void modifyGeometry();
    private:
       void init();
    private:
       GMergedMesh* create();
    private:
       GMergedMesh* combineSolids( std::vector<GSolid*>& solids);

       void loadCSG(const char* csgpath, std::vector<GSolid*>& solids );
       GMergedMesh* createPmtInBox();
       void createBoxInBox(std::vector<GSolid*>& solids);

       GMergedMesh* loadPmt();
    private:
       Opticks*         m_ok ; 
       NLODConfig*      m_lodconfig ; 
       int              m_lod ; 
       GGeoTestConfig*  m_config ; 
       GGeo*            m_ggeo ; 
       GGeoLib*         m_geolib ; 
       GBndLib*         m_bndlib ; 
       GMaker*          m_maker ; 
       unsigned int     m_verbosity ;

};


