#pragma once

struct NLODConfig ; 
class NCSGList ; 
class Opticks ; 
class OpticksResource ; 
class OpticksEvent ; 

class NGeoTestConfig ; 
class GGeoBase ; 

class GGeoLib ; 
class GMaterialLib ; 
class GSurfaceLib ; 
class GBndLib ; 
class GPmtLib ;   // <-- aim to remove
class GMeshLib ; 
class GNodeLib ; 

class GMaker ; 
class GMergedMesh ; 
class GVolume ; 
class GVolumeList ; 

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
       static const char* UNIVERSE_PV ; 
       static const char* UNIVERSE_LV ; 
    public:
       // testing utilities used from okg-/OpticksHubTest
       static const char* MakeArgForce(const char* funcname, const char* extra=NULL);
       static std::string MakeArgForce_(const char* funcname, const char* extra);
       static std::string MakeTestConfig_(const char* funcname);
    public:
       GGeoTest(Opticks* ok, GGeoBase* basis=NULL);
       int getErr() const ; 
    private:
       void init();
       GMergedMesh* initCreateCSG();
       GMergedMesh* initCreateBIB();
       void setErr(int err); 
    public:
       // GGeoBase

       GScintillatorLib* getScintillatorLib() const ;
       GSourceLib*       getSourceLib() const ;
       GSurfaceLib*      getSurfaceLib() const ;
       GMaterialLib*     getMaterialLib() const ;
       GMeshLib*         getMeshLib() const ;

       GBndLib*          getBndLib() const ;    
       GPmtLib*          getPmtLib() const ;
       GGeoLib*          getGeoLib() const ;
       GNodeLib*         getNodeLib() const ;

       const char*       getIdentifier() const ;
       GMergedMesh*      getMergedMesh(unsigned index) const ;

    private:
       void autoTestSetup(NCSGList* csglist);
       void relocateSurfaces(GVolume* solid, const char* spec) ;
       void reuseMaterials(NCSGList* csglist);
       void reuseMaterials(const char* spec);
    public:
       void dump(const char* msg="GGeoTest::dump");
    public:
       NGeoTestConfig* getConfig();
       NCSGList*       getCSGList() const ;
       NCSG*           getUniverse() const ;
       NCSG*           findEmitter() const ;
       unsigned        getNumTrees() const ;
       NCSG*           getTree(unsigned index) const ;
    public:
       GVolumeList*     getVolumeList();
    public:
       void anaEvent(OpticksEvent* evt);
    private:
       GMergedMesh* combineVolumes( std::vector<GVolume*>& volumes, GMergedMesh* mm0);
       GVolume*      makeVolumeFromConfig( unsigned i );
       void         importCSG(std::vector<GVolume*>& volumes );

       void         createBoxInBox(std::vector<GVolume*>& volumes);
       GMergedMesh* createPmtInBox();

       void labelPartList( std::vector<GVolume*>& volumes );

    private:
       Opticks*         m_ok ; 
       const char*      m_config_ ; 
       NGeoTestConfig*  m_config ; 
       unsigned         m_verbosity ;
       OpticksResource* m_resource ; 
       bool             m_dbgbnd ; 
       bool             m_dbganalytic ; 
       NLODConfig*      m_lodconfig ; 
       int              m_lod ; 
       bool             m_analytic ; 
       const char*      m_csgpath ; 
       bool             m_test ; 
    private:
       // base geometry and stolen libs 
       GGeoBase*        m_basis ; 
       GPmtLib*         m_pmtlib ; 
       GMeshLib*        m_meshlib ; 
   private:
       // local resident libs
       GMaterialLib*    m_mlib ; 
       GSurfaceLib*     m_slib ; 
       GBndLib*         m_bndlib ;  
       GGeoLib*         m_geolib ; 
       GNodeLib*        m_nodelib ; 
    private:
       // actors
       GMaker*          m_maker ; 
       NCSGList*        m_csglist ; 
       GVolumeList*      m_solist ; 
       int              m_err ; 

};


