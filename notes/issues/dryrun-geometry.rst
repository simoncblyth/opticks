dryrun-geometry
================


* want to dryrun geometry conversion prior to getting GPU involved in OGeo::convert 
* geometry methods too spread around

  * suspect will have difficultly doing this at GGeo level  



::

     190 OpticksHub::OpticksHub(Opticks* ok)
     191    :
     ...
     197    m_geometry(NULL),
     198    m_ggeo(GGeo::GetInstance()),   // if there is a GGeo instance already extant adopt it, otherwise load one  
     199    m_gscene(NULL),
     200    m_composition(new Composition(m_ok)),
     ...
     216 {
     217    init();
     219 }
     220 


     227 void OpticksHub::init()
     228 {   
     ...
     251     if( m_ggeo == NULL )
     252     {   
     253         loadGeometry() ;
     254     }
     255     else
     256     {   
     257         adoptGeometry() ;
     258     }
     259     if(m_err) return ;
     262     
     263     configureGeometry() ;
     264     
     265     deferredGeometryPrep();
     266 
     267 


     539 void OpticksHub::loadGeometry()
     540 {
     541     assert(m_geometry == NULL && "OpticksHub::loadGeometry should only be called once");
     542 
     543     LOG(info) << "[ " << m_ok->getIdPath()  ;
     544 
     545     m_geometry = new OpticksGeometry(this);   // m_lookup is set into m_ggeo here 
     546 
     547     m_geometry->loadGeometry();
     548 
     549     m_ggeo = m_geometry->getGGeo();
     550 
     551     m_gscene = m_ggeo->getScene();
     552 
     553 
     554     //   Lookup A and B are now set ...
     555     //      A : by OpticksHub::configureLookupA (ChromaMaterialMap.json)
     556     //      B : on GGeo loading in GGeo::setupLookup
     557 
     558 
     559     if(m_ok->isTest())  // --test : instanciate GGeoTest 
     560     {
     561         LOG(info) << "--test modifying geometry" ;
     562 
     563         assert(m_geotest == NULL);
     564 
     565         GGeoBase* basis = getGGeoBasePrimary(); // ana OR tri depending on --gltf
     566 
     567         m_geotest = createTestGeometry(basis);
     568 
     569         int err = m_geotest->getErr() ;
     570         if(err)
     571         {
     572             setErr(err);
     573             return ;
     574         }
     575     }
     576     else
     577     {
     578         LOG(LEVEL) << "NOT modifying geometry" ;
     579     }
     580 
     581     registerGeometry();
     582 
     583     m_ggeo->setComposition(m_composition);
     584 
     585     m_ggeo->close();  // mlib and slib  (June 2018, following remove the auto-trigger-close on getIndex in the proplib )
     586 
     587     LOG(info) << "]" ;
     588 }


