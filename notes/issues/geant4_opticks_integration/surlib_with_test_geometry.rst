SurLib with Test Geometry
===========================

The `--test` option results in a totally changed mesh0 
so actions done in GGeo::loadFromCache depending 
on specific geometry volume indices will go stale...

Idea for test geometry is to have all the materials
and just use simplified geometry. Problems are that 
surfaces via GSur wrappers mix properties with specific geometry:
 
* border surfaces need to use PV indices 
* skin surfaces need to use LV names

PMTInBox is the test geometry that most needs 
to have proper surface handling... 
Hmm getting the general case to work just for the cathode
surface seems overkill..

So stick with CTestDetector::kludgePhotoCathode for now.



::

    184 void OpticksGeometry::loadGeometry()
    185 {
    186     bool modify = m_opticks->hasOpt("test") ;
    ...
    190     loadGeometryBase();  //  usually from cache
    ...
    199     if(modify) modifyGeometry() ;
    200 


    256 void OpticksGeometry::modifyGeometry()
    257 {
    258     assert(m_opticks->hasOpt("test"));
    259     LOG(debug) << "OpticksGeometry::modifyGeometry" ;
    260 
    261     std::string testconf = m_fcfg->getTestConfig();
    262     m_ggeo->modifyGeometry( testconf.empty() ? NULL : testconf.c_str() );
    263 
    264 
    265     if(m_ggeo->getMeshVerbosity() > 2)
    266     {
    267         GMergedMesh* mesh0 = m_ggeo->getMergedMesh(0);
    268         if(mesh0)
    269         {
    270             mesh0->dumpSolids("OpticksGeometry::modifyGeometry mesh0");
    271             mesh0->save("$TMP", "GMergedMesh", "modifyGeometry") ;
    272         }
    273     }
    274 
    275 
    276     TIMER("modifyGeometry"); 
    277 }



::

     601 void GGeo::loadFromCache()
     602 {  
     603     LOG(trace) << "GGeo::loadFromCache START" ;
     604 
     605     m_geolib = GGeoLib::load(m_opticks);
     606 
     607     const char* idpath = m_opticks->getIdPath() ;
     608     m_meshindex = GItemIndex::load(idpath, "MeshIndex");
     609 
     610     if(m_volnames)
     611     {
     612         m_pvlist = GItemList::load(idpath, "PVNames");
     613         m_lvlist = GItemList::load(idpath, "LVNames");
     614     }
     615 
     616     m_bndlib = GBndLib::load(m_opticks);  // GBndLib is persisted via index buffer, not float buffer
     617 
     618    
     619 
     620     m_materiallib = GMaterialLib::load(m_opticks);
     621     m_surfacelib  = GSurfaceLib::load(m_opticks);
     622 
     623     m_bndlib->setMaterialLib(m_materiallib);
     624     m_bndlib->setSurfaceLib(m_surfacelib);
     625 
     626     m_scintillatorlib  = GScintillatorLib::load(m_opticks);
     627     m_sourcelib  = GSourceLib::load(m_opticks);
     628 
     629     m_surlib = new GSurLib(this) ;
     630     m_surlib->dump("GGeo::loadFromCache GSurLib::dump");
     631 
     632     LOG(trace) << "GGeo::loadFromCache DONE" ;
     633 }



