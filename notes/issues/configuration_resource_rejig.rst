configuration_resource_rejig
=================================

Its inconvenient that OpticksResource/BOpticksResource instantiation within Opticks instance
happens before Opticks::configure is called.  

* How disruptive would it be to move Opticks::initResource into the tail of Opticks::configure ?

  * it means some tests will need to call Opticks::configure in order to have access to resources


Overview
-----------

* the constraints of live updating over UDP (which is not currently used) led to the deferred configuration




Why is configuration deferred anyhow ?
-------------------------------------------

* twas for distributed config so that Cfg classes from higher level projects 
  get a chance to run their "addOptionS" etc before the commandline gets parsed 
  on invoking Opticks::configure

  * BUT : I note that most of the Cfg classes are in optickscore, possibly 
    the others could be moved there too ? Would that allow immediate configuration ?

* in pre-history Opticks ran a UDP server(or was that g4daeview.py) that could pick up commandlines during running,
  did that have any bearing on this ?

* there is some use of Listeners and boost::bind see BCfg : this is the reason the 
  listener instances from oglrap sych as Interactor, Scene and Renderer are not yet 
  instanciated (an might never be) so early in app launch

  * perhaps the Cfg classes themselves could listen, and the real classes use them
    via pointers 

* note that the live config machinery predates the adoption of ImGui which 
  mostly removed the need for live config over UDP

* note an alternative config apparoache is in use with BConfig (not BCfg)
  as used by NSceneConfig and various others 


Fails from moving Opticks::initResource into Opticks::configure after the commandline parse
---------------------------------------------------------------------------------------------

::

    FAILS:
      116/122 Test #116: NPYTest.NSceneConfigTest                      Child aborted***Exception:     0.09      caused by stricter BConfig, FIXED

      8  /26  Test #8  : OpticksCoreTest.OpticksFlagsTest              ***Exception: SegFault         0.07      FIXED : added ok.configure() 
      13 /26  Test #13 : OpticksCoreTest.OpticksResourceTest           ***Exception: SegFault         0.08      FIXED : added ok.configure()
      13 /50  Test #13 : GGeoTest.GScintillatorLibTest                 ***Exception: SegFault         0.07   
      15 /50  Test #15 : GGeoTest.GSourceLibTest                       ***Exception: SegFault         0.07      
      16 /50  Test #16 : GGeoTest.GBndLibTest                          ***Exception: SegFault         0.07   
      31 /50  Test #31 : GGeoTest.BoundariesNPYTest                    ***Exception: SegFault         0.07   
      33 /50  Test #33 : GGeoTest.GBBoxMeshTest                        ***Exception: SegFault         0.06   
      35 /50  Test #35 : GGeoTest.GFlagsTest                           ***Exception: SegFault         0.07             ditto    
      36 /50  Test #36 : GGeoTest.GGeoLibTest                          ***Exception: SegFault         0.08   
      37 /50  Test #37 : GGeoTest.GGeoTest                             Child aborted***Exception:     0.07   
      39 /50  Test #39 : GGeoTest.GMergedMeshTest                      ***Exception: SegFault         0.08   
      45 /50  Test #45 : GGeoTest.GSurfaceLibTest                      ***Exception: SegFault         0.06   
      47 /50  Test #47 : GGeoTest.NLookupTest                          ***Exception: SegFault         0.06   
      49 /50  Test #49 : GGeoTest.GSceneTest                           Child aborted***Exception:     0.33      FIXED :  UNRELATED probe assert 


      2  /19  Test #2  : OptiXRapTest.OScintillatorLibTest             ***Exception: SegFault         0.21      FIXED : ok.configure


      18 /19  Test #18 : OptiXRapTest.intersect_analytic_test          Child aborted***Exception:     1.24   
      19 /19  Test #19 : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.16      
      KNOWN quartic issue       
         

      3  /18  Test #3  : ExtG4Test.X4SolidTest                         ***Exception: SegFault         0.14     FIXED TOO
      11 /18  Test #11 : ExtG4Test.X4PhysicalVolumeTest                Child aborted***Exception:     0.15   
      12 /18  Test #12 : ExtG4Test.X4PhysicalVolume2Test               Child aborted***Exception:     0.15       
    [blyth@localhost opticks]$ 

 
Review Configuration
-----------------------

Almost all config is done in optickscore, only 4 Cfg classes at 
higher levels. Three of them are hooked up by OpticksViz::init the other GGeoCfg
is labelled as UDP only and appears not to be used anyhow.

* can those 4 be brought doen to the core ? 

::

    blyth@localhost opticks]$ find . -name '*Cfg.hh'
    ./boostrap/BCfg.hh

    ./optickscore/CameraCfg.hh
    ./optickscore/ClipperCfg.hh
    ./optickscore/CompositionCfg.hh
    ./optickscore/DemoCfg.hh
    ./optickscore/TrackballCfg.hh
    ./optickscore/ViewCfg.hh
    ./optickscore/OpticksCfg.hh

    ./ggeo/GGeoCfg.hh             
    ./oglrap/RendererCfg.hh        Renderer::PRINT
    ./oglrap/SceneCfg.hh           Scene::TARGET 
    ./oglrap/InteractorCfg.hh      Interactor::DRAGFACTOR Interaction::OPTIXMODE


RendererCfg
~~~~~~~~~~~~~

::

    192 void Renderer::configureI(const char* name, std::vector<int> values )
    193 {
    194     if(values.empty()) return ;
    195     if(strcmp(name, PRINT)==0) Print("Renderer::configureI");
    196 }



ViewCfg provides --eye/--look/--up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

optickscore/ViewCfg.hh::


     06 template <class Listener>
      7 class OKCORE_API ViewCfg : public BCfg {
      8 public:
      9    ViewCfg(const char* name, Listener* listener, bool live);
     10 };

optickscore/ViewCfg.cc::

     14 template <class Listener>
     15 ViewCfg<Listener>::ViewCfg(const char* name, Listener* listener, bool live) 
     16    : 
     17    BCfg(name, live) 
     18 {
     19        addOptionS<Listener>(listener, "eye", "Comma delimited eye position in model-extent coordinates, eg 0,0,-1  ");
     20        addOptionS<Listener>(listener, "look","Comma delimited look position in model-extent coordinates, eg 0,0,0  ");
     21        addOptionS<Listener>(listener, "up",  "Comma delimited up direction in model-extent frame, eg 0,1,0 " );
     22 }
     23 


UDP server
~~~~~~~~~~~~~

::

    [blyth@localhost optickscore]$ opticks-f UDP
    ./boostrap/brapdev.bash:     401     m_delegate->liveConnect(m_cfg); // hookup live config via UDP messages
    ./boostrap/brapdev.bash:When an external UDP message arrives each of the singletons liveline methods are 
    ./ggeo/GGeoCfg.hh:           "[UDP only], up to 4 comma delimited integers, eg 10,11,3158,0  \n"
    ./opticksgeo/OpticksHub.cc:        m_delegate->liveConnect(m_cfg); // hookup live config via UDP messages
    ./optickscore/CompositionCfg.cc:           "[UDP only], up to 4 comma delimited integers, eg:\n"
    ./optickscore/okc.bash:* reuse NumpyServer infrastructure for UDP messaging allowing live reconfig of objects 
    ./externals/imgui.bash:  would like everything to be doable from console and over UDP messaging 
    ./externals/glfw.bash:external events, eg messages from UDP,  ZeroMQ
    ./externals/cuda.bash:https://github.com/cudpp/cudpp/wiki/BuildingCUDPPWithMavericks
    ./thrustrap/thrap.bash:* https://github.com/cudpp/cudpp/wiki/BuildingCUDPPWithMavericks
    ./numpyserver/net_manager.hpp:       m_udp_server(m_local_io_service, delegate, delegate_io_service, delegate->getUDPPort()),
    ./numpyserver/numpydelegate.cpp:    if(strcmp(name, "udpport")==0) setUDPPort(values.back());
    ./numpyserver/numpydelegate.cpp:void numpydelegate::setUDPPort(int port)
    ./numpyserver/numpydelegate.cpp:int numpydelegate::getUDPPort()
    ./numpyserver/numpydelegate.hpp:   void setUDPPort(int port);
    ./numpyserver/numpydelegate.hpp:   int  getUDPPort();
    ./numpyserver/numpydelegateCfg.hpp:       addOptionI<Listener>(listener, "udpport",     "UDP Port");
    ./numpyserver/numpyserver.bash:Asynchronous ZMQ and UDP server with NPY serialization decoding.
    ./numpyserver/numpyserver.bash:that listens for UDP and ZMQ connections.
    ./numpyserver/numpyserver.bash:Other arguments identify the UDP port and ZMQ backend endpoint.
    ./numpyserver/numpyserver.bash:      Equivalent for UDP messages, with posts to delegate on_msg 
    ./numpyserver/numpyserver.bash:UDP Reply to sender
    ./numpyserver/numpyserver.bash:While numpyserver OR glfwtest is running, sendrecv UDP.::
    ./numpyserver/numpyserver.bash:UDP just send test::
    ./numpyserver/numpyserver.bash:     UDP_PORT=8080 udp.py hello world
    ./numpyserver/numpyserver.bash:UDP testing with reply:: 
    ./numpyserver/numpyserver.bash:      UDP_PORT=8080 udpr.py hello
    ./numpyserver/numpyserver.bash:    UDP_PORT=8080 udpr.py ${1:-hello_world} 
    ./numpyserver/numpyserver.hpp:   void send(std::string& addr, unsigned short port, std::string& msg );                   // "send" as UDP is connectionless
    ./numpyserver/tests/NumpyServerTest.cc:        UDP_PORT=13 udp.py hello
    [blyth@localhost opticks]$ 


::

     05 
      6 template <class Listener>
      7 class GGEO_API GGeoCfg : public BCfg {
      8 public:
      9    GGeoCfg(const char* name, Listener* listener, bool live) : BCfg(name, live)
     10    {
     11 
     12        addOptionS<Listener>(listener, Listener::PICKFACE,
     13            "[UDP only], up to 4 comma delimited integers, eg 10,11,3158,0  \n"
     14            "to target single face index 10 (range 10:11) of solid index 3158 in mesh index 0 \n"
     15            "\n"
     16            "    face_index0 \n"
     17            "    face_index1 \n"
     18            "    solid_index \n"
     19            "    mergedmesh_index  (currently only 0 non-instanced operational) \n"
     20            "\n"
     21            "see: GGeoCfg.hh\n"
     22            "     Composition::setPickFace\n"
     23            "     Scene::setFaceRangeTarget\n"
     24            "     GGeo::getFaceRangeCenterExtent\n"
     25       );
     26 
     27    }
     28 };


Doesnt look like its used, unless via BCfg base::

    [blyth@localhost opticks]$ opticks-f GGeoCfg
    ./ggeo/GGeoCfg.cc:#include "GGeoCfg.hh"
    ./ggeo/GGeoCfg.hh:class GGEO_API GGeoCfg : public BCfg {
    ./ggeo/GGeoCfg.hh:   GGeoCfg(const char* name, Listener* listener, bool live) : BCfg(name, live) 
    ./ggeo/GGeoCfg.hh:           "see: GGeoCfg.hh\n"
    ./ggeo/ggeodev.bash:GGeoCfg
    ./ggeo/GGeo.hh:        // see GGeoCfg.hh
    ./ggeo/CMakeLists.txt:    GGeoCfg.cc
    ./ggeo/CMakeLists.txt:    GGeoCfg.hh
    [blyth@localhost opticks]$ 



BCfg::

     18 #ifdef __clang__
     19 #pragma GCC visibility pop
     20 #endif
     21 
     22 /*
     23 Listener classes need to provide a methods::
     24 
     25    void configureF(const char* name, std::vector<float> values);
     26    void configureI(const char* name, std::vector<int> values);
     27    void configureS(const char* name, std::vector<std::string> values);
     28  
     29 which is called whenever the option parsing methods are called. 
     30 Typically the last value in the vector should be used to call the Listeners 
     31 setter method as selected by the name.
     32 */
     33 #include "BRAP_API_EXPORT.hh"
     34 #include "BRAP_HEAD.hh"
     35 
     36 
     37 #ifdef _MSC_VER
     38 // m_vm m_desc m_others m_commandline m_error_message needs dll-interface
     39 #pragma warning( disable : 4251 )
     40 #endif
     41 
     42 
     43 class BRAP_API BCfg {



Listeners and boost::bind
------------------------------

Adding options tees up boost bind notifier. Hmm was this how live updating (from UDP server) worked ?

::

    118 template <class Listener>
    119 inline void BCfg::addOptionF(Listener* listener, const char* name, const char* description )
    120 {
    121         m_desc.add_options()(name,
    122                              boost::program_options::value<std::vector<float> >()
    123                                 ->composing()
    124                                 ->notifier(boost::bind(&Listener::configureF, listener, name, _1)),
    125                              description) ;
    126 }
    127 
    128 template <class Listener>
    129 inline void BCfg::addOptionI(Listener* listener, const char* name, const char* description )
    130 {
    131         m_desc.add_options()(name,
    132                              boost::program_options::value<std::vector<int> >()
    133                                 ->composing()
    134                                 ->notifier(boost::bind(&Listener::configureI, listener, name, _1)),
    135                              description) ;
    136 }
    137 
    138 
    139 template <class Listener>
    140 inline void BCfg::addOptionS(Listener* listener, const char* name, const char* description )
    141 {
    142         if(m_verbose)
    143         {
    144              printf("BCfg::addOptionS %s %s \n", name, description);
    145         }
    146         m_desc.add_options()(name,
    147                              boost::program_options::value<std::vector<std::string> >()
    148                                 ->composing()
    149                                 ->notifier(boost::bind(&Listener::configureS, listener, name, _1)),
    150                              description) ;
    151 }
    152 
    153 
    154 #include "BRAP_TAIL.hh"





::

    101 void OpticksViz::init()
    102 {
    103     m_hub->setCtrl(this);  // For "command(char* ctrl)" interface from lower levels to route via OpticksViz
    104 
    105     const char* shader_dir = getenv("OPTICKS_SHADER_DIR");
    106     const char* shader_incl_path = getenv("OPTICKS_SHADER_INCL_PATH");
    107     const char* shader_dynamic_dir = getenv("OPTICKS_SHADER_DYNAMIC_DIR");
    108     // envvars normally not defined, using cmake configure_file values instead
    109 
    110     m_scene      = new Scene(m_hub, shader_dir, shader_incl_path, shader_dynamic_dir ) ;
    111     m_frame       = new Frame ;
    112     m_interactor  = new Interactor(m_composition) ;
    113 
    114     m_interactor->setFrame(m_frame);
    115     m_interactor->setScene(m_scene);
    116     //m_interactor->setComposition(m_composition);
    117 
    118     m_scene->setInteractor(m_interactor);
    119 
    120     m_frame->setInteractor(m_interactor);
    121     m_frame->setComposition(m_composition);
    122     m_frame->setScene(m_scene);
    123 
    124     m_hub->add(new SceneCfg<Scene>(           "scene",       m_scene,                      true));
    125     m_hub->add(new RendererCfg<Renderer>(     "renderer",    m_scene->getGeometryRenderer(), true));
    126     m_hub->add(new InteractorCfg<Interactor>( "interactor",  m_interactor,                 true));
    127 



