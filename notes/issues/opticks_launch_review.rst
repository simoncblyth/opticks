Opticks Launch Review
=======================

High level review of the launching of an Opticks executable
such as "OKTest" identifying the packages and primary 
classes driving the action.


Issues 
--------

* OpticksResource (resident of Opticks) currently requires a geometry source .dae file 
  to exist (opticksdata) 

  * with new geometry that will not be the case, so currently an unrelated
    other geometry needs to fill the spot before the export can be done
    (historically export was done separately : as so it was always available)
    
    * so Opticks cannot currently run without opticksdata (even when it doesnt need to use it)

* to change from the dayabay default required envvar gymnastics 
  
* OpticksResource assumes a small number of harcoded geometries for naming purposes
 

Fix? How ? 
~~~~~~~~~~~

* split off specific geometry aspects of OpticksResource (the installpath related setup, eg installcache, can remain) 
  into a separate class that gets instanciated later (perhaps with OpticksHub?)

  * started this by pulling BResource out of BOpticksResource


OKTest
--------

OKTest::

    OKMgr ok(argc, argv);
    ok.propagate();         // m_run, m_propagator
    ok.visualize();         // m_viz


OKMgr instanciation
------------------------

::

    OKMgr::OKMgr(int argc, char** argv, const char* argforced ) 
        :
        m_log(new SLog("OKMgr::OKMgr")),
        m_ok(new Opticks(argc, argv, argforced)),         
        m_hub(new OpticksHub(m_ok)),            // immediate configure and loadGeometry 
        m_idx(new OpticksIdx(m_hub)),
        m_num_event(m_ok->getMultiEvent()),     // after hub instanciation, as that configures Opticks
        m_gen(m_hub->getGen()),
        m_run(m_hub->getRun()),
        m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),
        m_propagator(new OKPropagator(m_hub, m_idx, m_viz)),
        m_count(0)
    {
        init();
        (*m_log)("DONE");
    }


okc.Opticks::Opticks
----------------------

::

     410 void Opticks::init()
     411 {
     ...
     426     m_resource = new OpticksResource(this, m_envprefix, m_lastarg);
     427 
     428     setDetector( m_resource->getDetector() );
     ...
     431 }
    

Opticks validity from m_resource::

    1821 bool Opticks::isValid() {   return m_resource->isValid(); }
  
     // setInvalid is called when no daepath 

::

      77 OpticksResource::OpticksResource(Opticks* opticks, const char* envprefix, const char* lastarg)
      78     :
      79        BOpticksResource(envprefix),
      80        m_opticks(opticks),
      81        m_lastarg(lastarg ? strdup(lastarg) : NULL),
      82 
      83        m_geokey(NULL),
      84 
      85        m_query_string(NULL),

::

     254 void OpticksResource::init()
     255 {
     256    LOG(trace) << "OpticksResource::init" ;
     257 
     258    BStr::split(m_detector_types, "GScintillatorLib,GMaterialLib,GSurfaceLib,GBndLib,GSourceLib", ',' );
     259    BStr::split(m_resource_types, "GFlags,OpticksColors", ',' );
     260 
     261    readG4Environment();
     262    readOpticksEnvironment();
     263    readEnvironment();
     264 
     265    readMetadata();
     266    identifyGeometry();   // this assumes a small number hardcoded geometries 
     267    assignDetectorName();
     268    assignDefaultMaterial();
     269 
     270    LOG(trace) << "OpticksResource::init DONE" ;
     271 }
     272 


OpticksResource::readEnvironment asserts when no daepath is configured via the envvar mechanism::

     496 void OpticksResource::readEnvironment()
     497 {
     ...
     521     m_geokey = SSys::getenvvar(m_envprefix, "GEOKEY", DEFAULT_GEOKEY);
     522     const char* daepath = SSys::getenvvar(m_geokey);
     ...
     565     assert(daepath);
     566 
     567     setupViaSrc(daepath, query_digest.c_str());  // this sets m_idbase, m_idfold, m_idname done in base BOpticksResource
     568 
     569     assert(m_idpath) ;
     570     assert(m_idname) ;
     571     assert(m_idfold) ;
     572 }


     
Geometry Loading 
-----------------

* :doc:`geometry_review`


