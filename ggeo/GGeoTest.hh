#pragma once

struct NLODConfig ; 
class NCSGList ; 
class Opticks ; 
class OpticksResource ; 
class OpticksEvent ; 

class GGeoTestConfig ; 
class GGeoBase ; 
class GGeoLib ; 
class GBndLib ; 
class GPmtLib ; 
class GNodeLib ; 
class GMaker ; 
class GMergedMesh ; 
class GSolid ; 
class GSolidList ; 

/**
GGeoTest
=========

Creates simple test geometries from a commandline specification.

Canonical instance m_geotest resides in OpticksHub and is instanciated
only when the --test geometry option is active.  This happens
after standard geometry is loaded via OpticksHub::modifyGeometry.

Rejig
-------

* GGeoTest is now a GGeoBase subclass (just like GGeo and GScene)

* GGeoTest now has its own GGeoLib, to avoid dirty modifyGeometry
  appropach which cleared the basis mm



**/

#include "GGeoBase.hh"
#include "GGEO_API_EXPORT.hh"
class GGEO_API GGeoTest : public GGeoBase {
    public:
       GGeoTest(Opticks* ok, GGeoBase* basis=NULL);
    private:
       void init();
    public:
       // GGeoBase : constituents locally customized
       const char*       getIdentifier();
       GMergedMesh*      getMergedMesh(unsigned index);
       GGeoLib*          getGeoLib();
       GNodeLib*         getNodeLib();

       // GGeoBase : constituents passed along from basis geometry 
       GScintillatorLib* getScintillatorLib() ;
       GSourceLib*       getSourceLib() ;
       GPmtLib*          getPmtLib() ;
       GBndLib*          getBndLib() ;    // <-- maybe this one should be cloned/modified wrt base ?
    public:
       void dump(const char* msg="GGeoTest::dump");
    public:
       GGeoTestConfig* getConfig();
       NCSGList*       getCSGList() const ;
       NCSG*           findEmitter() const ;
       unsigned        getNumTrees() const ;
       NCSG*           getTree(unsigned index) const ;
    public:
       GSolidList*     getSolidList();
    public:
       void anaEvent(OpticksEvent* evt);
    private:
       GMergedMesh* create();
       GMergedMesh* combineSolids( std::vector<GSolid*>& solids, GMergedMesh* mm0);
       GSolid*      makeSolidFromConfig( unsigned i );
       void         loadCSG(const char* csgpath, std::vector<GSolid*>& solids );

       void         createBoxInBox(std::vector<GSolid*>& solids);
       GMergedMesh* createPmtInBox();

       void labelPartList( std::vector<GSolid*>& solids );

    private:
       Opticks*         m_ok ; 
       GGeoTestConfig*  m_config ; 
       OpticksResource* m_resource ; 
       bool             m_dbganalytic ; 
       NLODConfig*      m_lodconfig ; 
       int              m_lod ; 
       bool             m_analytic ; 
       bool             m_test ; 
    private:
       // base geometry and libs taken from it
       GGeoBase*        m_basis ; 
       GBndLib*         m_bndlib ; 
       GPmtLib*         m_pmtlib ; 
    private:
       // locally created libs
       GGeoLib*         m_geolib ; 
       GNodeLib*        m_nodelib ; 
    private:
       // actors
       GMaker*          m_maker ; 
       NCSGList*        m_csglist ; 
       GSolidList*      m_solist ; 
       unsigned int     m_verbosity ;

};


