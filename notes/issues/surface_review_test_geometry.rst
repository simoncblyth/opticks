Surface Review Test Geometry
=================================

Problem with surfaces in test geometry
----------------------------------------

Cause of confusion is the conflation of two things:

* surface properties
* surface location (specified by sslv/bpv1/bpv2)

With test geometry wish to reuse some surface properties, 
from the base geometry but need to totally change surface location
to suit the boundary spec coming down the pipe (from NCSGList).

So need to derive a separate GSurfaceLib from scratch that is able to 
draw from the basis one for surface properties. This implies 
creating a from scratch GBndLib too.

Added some methods to GSurfaceLib to allow passing props from basis into 
a new lib with different locations::   

     89         // methods to assist with de-conflation of surface props and location
     90         void addBorderSurface(GPropertyMap<float>* surf, const char* pv1, const char* pv2);
     91         void addSkinSurface(GPropertyMap<float>* surf, const char* sslv_ );


How to handle test geometry in CSurfaceLib::convert ?
------------------------------------------------------------

**Best way** 
    prepare the GSurfaceLib in a manner such that CSurfaceLib 
    doesnt need to know if test/full geometry.


GGeoTest : GMaterialLib from base + AbInitio GBndLib/GSurfaceLib  
---------------------------------------------------------------------

* dev in GBndLibInitTest 
* How to handle surface indices in the bndlib ? GBndLib buffers are dynamic to handle added surfaces, so may just work ?

* from OptiX point of view (GPU geometry) all thats needed is the GBndLib to create the texture 

::
 
    141 GSolid* GMaker::makeFromCSG(NCSG* csg, GBndLib* bndlib, unsigned verbosity )
    142 {
    ...
    160     GSolid* solid = new GSolid(index, transform, mesh, UINT_MAX, NULL );
    161 
    162     // csg is mesh-qty not a node-qty, boundary spec is a node-qty : so this is just for testing
    163 
    164     unsigned boundary = bndlib->addBoundary(spec);  // only adds if not existing
    165 
    166     solid->setBoundary(boundary);     // unlike ctor these create arrays
    167 
    168     solid->setSensor( NULL );
    169 
    170 
    171     OpticksCSG_t type = csg->getRootType() ;
    172 
    173     const char* shapename = CSGName(type);
    174     std::string lvn = GMaker::LVName(shapename, index);
    175     std::string pvn = GMaker::PVName(shapename, index);
    176 
    177     solid->setPVName( strdup(pvn.c_str()) );
    178     solid->setLVName( strdup(lvn.c_str()) );
    179     solid->setCSGFlag( type );
    180 
    181     GParts* pts = GParts::make( csg, spec, verbosity );
    182 
    183 
    184     solid->setParts( pts );


::

    simon:opticks blyth$ opticks-find GGeoTest | grep new
    ./ggeo/GGeoTest.cc:    m_config(new GGeoTestConfig(ok->getTestConfig())),
    ./ggeo/tests/GGeoTestConfigTest.cc:    GGeoTestConfig* gtc = new GGeoTestConfig(CONFIG);
    ./opticksgeo/OpticksHub.cc:    GGeoTest* testgeo = new GGeoTest(m_ok, basis);
    simon:opticks blyth$ 


::

    295 void OpticksHub::loadGeometry()
    296 {
    297     assert(m_geometry == NULL && "OpticksHub::loadGeometry should only be called once");
    298 
    299     LOG(info) << "OpticksHub::loadGeometry START" ;
    300 
    301     m_geometry = new OpticksGeometry(this);   // m_lookup is set into m_ggeo here 
    302 
    303     m_geometry->loadGeometry();
    304 
    305 
    306     //   Lookup A and B are now set ...
    307     //      A : by OpticksHub::configureLookupA (ChromaMaterialMap.json)
    308     //      B : on GGeo loading in GGeo::setupLookup
    309 
    310     m_ggeo = m_geometry->getGGeo();
    311     m_gscene = m_ggeo->getScene();
    312 
    313     if(m_ok->isTest())
    314     {
    315         LOG(info) << "OpticksHub::loadGeometry --test modifying geometry" ;
    316 
    317         assert(m_geotest == NULL);
    318 
    319         GGeoBase* basis = getGGeoBase(); // ana OR tri depending on --gltf
    320 
    321         m_geotest = createTestGeometry(basis);
    322     }
    323     else


    339 GGeoTest* OpticksHub::createTestGeometry(GGeoBase* basis)
    340 {
    341     assert(m_ok->isTest());
    342 
    343     LOG(info) << "OpticksHub::createTestGeometry START" ;
    344 
    345     GGeoTest* testgeo = new GGeoTest(m_ok, basis);
    346 
    347     LOG(info) << "OpticksHub::createTestGeometry DONE" ;
    348 
    349     return testgeo ;
    350 }





* GMaker::makeFromCSG assigns PV, LV names to solids
* GGeoTest collects solids into GNodeLib 




