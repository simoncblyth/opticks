bcfg-rel(){      echo boostrap ; }
bcfg-src(){      echo $(bcfg-rel)/bcfg.bash ; }
bcfg-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(bcfg-src)} ; }
bcfg-vi(){       vi $(bcfg-source) ; }
bcfg-usage(){ cat << EOU

Boost Bind Based Configuration
================================

NB the NConfigurable base class approach may 
be a simpler alternative, as it allows all NConfigurable
sub-classes to be treated via same API. Meaning that 
can avoid the templated methods that cause create pain
to get past compilers.



Config code formerly lived at boost/bpo/bcfg, 
now consolidated into boostrap aka brap-.

main::

     357 int App::config(int argc, char** argv)
     358 {
     359     m_cfg  = new Cfg("unbrella", false) ;
     360     m_fcfg = new FrameCfg<Frame>("frame", m_frame,false);
     361     m_cfg->add(m_fcfg);
     362 #ifdef NPYSERVER
     363     m_cfg->add(new numpydelegateCfg<numpydelegate>("numpydelegate", m_delegate, false));
     364 #endif
     365 
     366     m_cfg->add(new SceneCfg<Scene>(           "scene",       m_scene,                      true));
     367     m_cfg->add(new RendererCfg<Renderer>(     "renderer",    m_scene->getGeometryRenderer(), true));
     368     m_cfg->add(new InteractorCfg<Interactor>( "interactor",  m_interactor,                 true));
     369 
     370     m_composition->addConfig(m_cfg);
     371 
     372     m_cfg->commandline(argc, argv);
     ...
     399 
     400 #ifdef NPYSERVER
     401     m_delegate->liveConnect(m_cfg); // hookup live config via UDP messages
     402     m_delegate->setNumpyEvt(m_evt); // allows delegate to update evt when NPY messages arrive
     403     m_server = new numpyserver<numpydelegate>(m_delegate); // connect to external messages 
     404 #endif

Each constituent cfg has type and instance of long lived singleton objects
When an external UDP message arrives each of the singletons liveline methods are 
called with the string::

    095 void numpydelegate::interpretExternalMessage(std::string msg)
     96 {
     97     //printf("numpydelegate::interpretExternalMessage %s \n", msg.c_str());
     98     for(size_t i=0 ; i<m_live_cfg.size() ; ++i)
     99     {
    100         Cfg* cfg = m_live_cfg[i];
    101         cfg->liveline(msg.c_str());  // cfg listener objects may get live configured 
    102     }
    103 }

Only leaf cfg can be configured::

    108 void Cfg::liveline(const char* _line)
    109 {
    110     if(m_others.empty())
    111     {
    112         std::vector<std::string> unrecognized = parse_liveline(_line);


Boost command line program options parsing is applied to the liveline tokens::

    194 std::vector<std::string> Cfg::parse_tokens(std::vector<std::string>& tokens)
    195 {
    196     std::vector<std::string> unrecognized ;
    197 #ifdef VERBOSE
    198     dump(tokens, "Cfg::parse_tokens input");
    199 #endif
    200 
    201     po::command_line_parser parser(tokens);
    202     parser.options(m_desc);
    203     parser.allow_unregistered();
    204     po::parsed_options parsed = parser.run();
    205 
    206 #ifdef VERBOSE
    207     dump(parsed, "Cfg::parse_tokens parsed");
    208 #endif
    209     po::store(parsed, m_vm);
    210     po::notify(m_vm);
    211 
    212     std::vector<std::string> unrec = po::collect_unrecognized(parsed.options, po::include_positional);
    213     unrecognized.assign(unrec.begin(), unrec.end());
    214 
    215     return unrecognized ;
    216 }


CameraCfg.hh specializes Cfg with specific options::

     01 #pragma once
      2 #include "Cfg.hh"
      3 
      4 template <class Listener>
      5 class CameraCfg : public Cfg {
      6 public:
      7    CameraCfg(const char* name, Listener* listener, bool live) : Cfg(name, live)
      8    {
      9        addOptionI<Listener>(listener, Listener::PRINT,    "Print");
     10 
     11        addOptionF<Listener>(listener, Listener::YFOV,     "Vertical Field of view in degrees");
     12        addOptionF<Listener>(listener, Listener::NEAR,     "Near distance");
     13        addOptionF<Listener>(listener, Listener::FAR,      "Far distance" );
     14        addOptionF<Listener>(listener, Listener::PARALLEL, "Parallel or perspective");
     15    }
     16 };

CameraCfg is instanciated with Camera class as listener and m_camera as instance::

    169 void Composition::addConfig(Cfg* cfg)
    170 {
    171     // hmm problematic with bookmarks that swap out Camera, View, ...
    172     cfg->add(new CompositionCfg<Composition>("composition", this,          true));
    173     cfg->add(new CameraCfg<Camera>(          "camera",      getCamera(),   true));
    174     cfg->add(new ViewCfg<View>(              "view",        getView(),     true));
    175     cfg->add(new TrackballCfg<Trackball>(    "trackball",   getTrackball(),true));
    176     cfg->add(new ClipperCfg<Clipper>(        "clipper",     getClipper(),  true));
    177 }

Mechanics of notification handled in Cfg with boost bind notifiers which dispatch to the configureF/S/I methods::

    113 template <class Listener>
    114 void Cfg::addOptionF(Listener* listener, const char* name, const char* description )
    115 {
    116         m_desc.add_options()(name,
    117                              boost::program_options::value<std::vector<float> >()
    118                                 ->composing()
    119                                 ->notifier(boost::bind(&Listener::configureF, listener, name, _1)),
    120                              description) ;
    121 }


Requirements for listeners

* name options in static cstr
* implement configureF configureI configureS as appropriate 
  that handle the arrivals, typically changing a parameter of the listener object 

::

     10 class Camera : public NConfigurable  {
     11   public:
     12     
     13      static const char* PRINT ;
     14      static const char* NEAR ; 
     15      static const char* FAR ; 
     16      static const char* YFOV ;
     17      static const char* PARALLEL ;

EOU
}


bcfg-sdir(){ echo $(opticks-home)/$(bcfg-rel) ; }
bcfg-env(){      elocal- ; opticks- ;  }

bcfg-idir(){ echo $(opticks-idir) ; }
bcfg-bdir(){ echo $(opticks-bdir)/$(bcfg-rel) ; }

bcfg-scd(){  cd $(bcfg-sdir); }
bcfg-cd(){  cd $(bcfg-sdir); }

bcfg-icd(){  cd $(bcfg-idir); }
bcfg-bcd(){  cd $(bcfg-bdir); }

bcfg-name(){ echo CfgTest ; }

bcfg-wipe(){
   local bdir=$(bcfg-bdir)
   rm -rf $bdir
}

bcfg-txt(){ vi $(bcfg-sdir)/CMakeLists.txt ; }

bcfg-make(){
   local iwd=$PWD

   bcfg-bcd
   make $*

   cd $iwd
}

bcfg-install(){
   bcfg-make install
}

bcfg-bin(){ echo $(bcfg-idir)/bin/$(bcfg-name) ; }

bcfg-export()
{
   echo -n
}

bcfg-run(){
   local bin=$(bcfg-bin)
   bcfg-export
   $bin $*
}

bcfg-runq(){
   local bin=$(bcfg-bin)
   bcfg-export

   local parms=""
   local p
   for p in "$@" ; do
      [ "${p/ /}" == "$p" ] && parms="${parms} $p" || parms="${parms} \"${p}\""
   done

   cat << EOC  | sh 
   $bin $parms
EOC
}


bcfg--()
{
    bcfg-make clean
    bcfg-make 
    bcfg-install
}




