#pragma once

struct NLODConfig ; 
class NCSGList ; 
class Opticks ; 
class OpticksEvent ; 

class GGeoTestConfig ; 
class GGeoBase ; 
class GGeoLib ; 
class GBndLib ; 
class GPmtLib ; 
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
       GGeoTest(Opticks* ok, GGeoTestConfig* config, GGeoBase* ggeobase=NULL);
       void dump(const char* msg="GGeoTest::dump");
       void modifyGeometry();

       unsigned getNumTrees();
       NCSG* getTree(unsigned index);
       void anaEvent(OpticksEvent* evt);

    private:
       void init();
    private:
       GMergedMesh* create();
    private:
       GMergedMesh* combineSolids( std::vector<GSolid*>& solids, GMergedMesh* mm0);
       GSolid* makeSolidFromConfig( unsigned i );
       void loadCSG(const char* csgpath, std::vector<GSolid*>& solids );

       void createBoxInBox(std::vector<GSolid*>& solids);
       GMergedMesh* createPmtInBox();

       void labelPartList( std::vector<GSolid*>& solids );

    private:
       Opticks*         m_ok ; 
       bool             m_dbganalytic ; 
       NLODConfig*      m_lodconfig ; 
       int              m_lod ; 
       GGeoTestConfig*  m_config ; 
       GGeoBase*        m_ggeobase ; 
       GGeoLib*         m_geolib ; 
       GBndLib*         m_bndlib ; 
       GPmtLib*         m_pmtlib ; 
       GMaker*          m_maker ; 
       NCSGList*        m_csglist ; 
       unsigned int     m_verbosity ;

};


