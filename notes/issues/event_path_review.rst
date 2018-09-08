event_path_review
==============================

Context
----------

* :doc:`ckm_cerenkov_generation_align`


WIP : For direct workflow : would be more convenient to save events within the keydir 
------------------------------------------------------------------------------------------

* especially important for passing gensteps between executables, for direct running
  can have a standard path within the keydir for each tag at which to look for gensteps

  * first geometry + genstep collecting and writing executable is special : it should write its event
    and genstep into distinctive "standard" directory (perhaps under the name "source") within the
    geocache keydir 

  * all other executables sharing the same keydir can put their events underneath 
    a relpath named after the executable  
   

event categories : still needed in direct approach ?
--------------------------------------------------------

det 
    always g4live for direct approach, so makes no sense to have a det label, 
    effectively the keydir as pointed to by OPTICKS_KEY is the way to switch 
    between detector geometries

typ
    in old approach the typ came from commandline arguments like --torch (default) 
    --cerenkov --scintillation --natural that picked between fabricated 
    or opticksdata saved gensteps

    * in direct approach the gensteps collected from "executable-0" programatically defines the typ

    * HMM : in principal could change the gensteps, eg by enabling scintillation vs cerenkov 
      processes while using same geometry, hence same geocache.

    * but cannot assume easy access to commandline in direct approach 

      If are sharing same geocache between such different genstep typ then need to keep the typ
      category. 

tag
    needed for multiple events, and distinguishing G4 from OK events by negation
 


::

    epsilon:1 blyth$ ckm-kcd
    epsilon:1 blyth$ find . -name 'gs.npy'
    ./source/evt/g4live/natural/1/gs.npy
    epsilon:1 blyth$ 


::

    1072 const char* Opticks::getDirectGenstepPath() const { return m_resource->getDirectGenstepPath() ; }
    1073 const char* Opticks::getDirectPhotonsPath() const { return m_resource->getDirectPhotonsPath() ; }




former non-scalable approach
-----------------------------

* non-scalable because there is no tag or typ specification just the gensteps directly in geocache 

* HMM : is anything more than tag actually needed in direct route ?


directgensteppath getDirectGenstep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    epsilon:source blyth$ opticks-find directgensteppath 
    ./boostrap/BOpticksResource.cc:    m_directgensteppath(NULL),
    ./boostrap/BOpticksResource.cc:    m_directgensteppath = makeIdPathPath("directgenstep.npy");  
    ./boostrap/BOpticksResource.cc:    m_res->addPath("directgensteppath", m_directgensteppath ); 
    ./boostrap/BOpticksResource.cc:const char* BOpticksResource::getDirectGenstepPath() const { return m_directgensteppath ; } 
    ./boostrap/BOpticksResource.hh:       const char* m_directgensteppath ; 
    epsilon:opticks blyth$ 

    epsilon:opticks blyth$ opticks-find getDirectGenstepPath
    ./g4ok/G4Opticks.cc:    const char* gspath = m_ok->getDirectGenstepPath(); 
    ./optickscore/Opticks.cc:const char* Opticks::getDirectGenstepPath() const { return m_resource->getDirectGenstepPath() ; } 
    ./optickscore/Opticks.cc:    const char* path = getDirectGenstepPath();
    ./optickscore/Opticks.cc:    std::string path = getDirectGenstepPath();
    ./boostrap/BOpticksResource.cc:const char* BOpticksResource::getDirectGenstepPath() const { return m_directgensteppath ; } 
    ./optickscore/Opticks.hh:       const char*          getDirectGenstepPath() const ; 
    ./boostrap/BOpticksResource.hh:       const char* getDirectGenstepPath() const ;
    epsilon:opticks blyth$ 

::

    207 /**
    208 G4Opticks::propagateOpticalPhotons
    209 -----------------------------------
    210 
    211 Invoked from EventAction::EndOfEventAction
    212 
    213 TODO: relocate direct events inside the geocache ? 
    214       and place these direct gensteps and genphotons 
    215       within the OpticksEvent directory 
    216 
    217 
    218 **/
    219 
    220 int G4Opticks::propagateOpticalPhotons()
    221 {
    222     m_gensteps = m_genstep_collector->getGensteps();
    223     const char* gspath = m_ok->getDirectGenstepPath();
    224 
    225     LOG(info) << " saving gensteps to " << gspath ;
    226     m_gensteps->setArrayContentVersion(G4VERSION_NUMBER);
    227     m_gensteps->save(gspath);
    228 

    1898 bool Opticks::existsDirectGenstepPath() const
    1899 {
    1900     const char* path = getDirectGenstepPath();
    1901     return path ? BFile::ExistsFile(path) : false ;
    1902 }
    1903 

    epsilon:optickscore blyth$ opticks-find loadDirectGenstep
    ./opticksgeo/OpticksGen.cc:    m_direct_gensteps(m_ok->existsDirectGenstepPath() ? m_ok->loadDirectGenstep() : NULL ),
    ./optickscore/Opticks.cc:NPY<float>* Opticks::loadDirectGenstep() const 
    ./optickscore/Opticks.hh:       NPY<float>*          loadDirectGenstep() const ;
    epsilon:opticks blyth$ 



OpticksGen auto sets the sourcecode based on existance of direct gensteps::

     28 OpticksGen::OpticksGen(OpticksHub* hub)
     29     :
     30     m_hub(hub),
     31     m_gun(new OpticksGun(hub)),
     32     m_ok(hub->getOpticks()),
     33     m_cfg(m_ok->getCfg()),
     34     m_ggb(hub->getGGeoBase()),
     35     m_blib(m_ggb->getBndLib()),
     36     m_lookup(hub->getLookup()),
     37     m_torchstep(NULL),
     38     m_fabstep(NULL),
     39     m_input_gensteps(NULL),
     40     m_csg_emit(hub->findEmitter()),
     41     m_emitter_dbg(false),
     42     m_emitter(m_csg_emit ? new NEmitPhotonsNPY(m_csg_emit, EMITSOURCE, m_ok->getSeed(), m_emitter_dbg, m_ok->getMaskBuffer()) : NULL ),
     43     m_input_photons(NULL),
     44     m_input_primaries(m_ok->existsPrimariesPath() ? m_ok->loadPrimaries() : NULL ),
     45     m_direct_gensteps(m_ok->existsDirectGenstepPath() ? m_ok->loadDirectGenstep() : NULL ),
     46     m_source_code(initSourceCode())
     47 {
     48     init() ;
     49 }
     50 
     51 Opticks* OpticksGen::getOpticks() const { return m_ok ; }
     52 std::string OpticksGen::getG4GunConfig() const { return m_gun->getConfig() ; }
     53 
     54 bool OpticksGen::hasInputPrimaries() const
     55 {
     56     return m_input_primaries != NULL ;
     57 }
     58 
     59 
     60 unsigned OpticksGen::initSourceCode() const
     61 {
     62     unsigned code = 0 ;
     63     if(m_direct_gensteps)
     64     {
     65         code = GENSTEPSOURCE ;
     66     } 
     67     else if(m_input_primaries)
     68     {
     69         code = PRIMARYSOURCE ;
     70     } 
     71     else if(m_emitter)





How to distinguish the special key creating executable from key reading ?
----------------------------------------------------------------------------

The distinguishing thing is the direct translation of geometry done in G4Opticks::translateGeometry
so perhaps a "keysource" flag option for  BOpticksKey::SetKey(keyspec)

* actually can auto-detect this from the exename of the key matching that of the current executable

::

    139 GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
    140 {
    141     const char* keyspec = X4PhysicalVolume::Key(top) ;
    142     BOpticksKey::SetKey(keyspec);
    143     LOG(error) << " SetKey " << keyspec  ;
    144 
    145     Opticks* ok = new Opticks(0,0, fEmbeddedCommandLine);  // Opticks instanciation must be after BOpticksKey::SetKey
    146 


How to allow other executables to access paths written by the keysource executable ?
---------------------------------------------------------------------------------------

Provide two dirs, which are the same for the KeySource case, so can always write to evtbase
and can read from srcevtbase.

::

    474     const char* user = SSys::username();
    475     m_srcevtbase = makeIdPathPath("evt", user, "source");
    476     m_res->addDir( "srcevtbase", m_srcevtbase );
    477 
    478     const char* exename = SAr::Instance->exename();
    479     m_evtbase = isKeySource() ? strdup(m_srcevtbase) : makeIdPathPath("evt", user, exename ) ;
    480     m_res->addDir( "evtbase", m_evtbase );


How to handle multiple users sharing a geocache ?
---------------------------------------------------

* could move current TMP /tmp/username/opticks into  keydir ?

  * using a username dir  


Event Path machinery 
----------------------

Slightly nasty NPY special cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NPY has special handling of quad-argument save::

     707 template <typename T>
     708 NPY<T>* NPY<T>::load(const char* tfmt, const char* source, const char* tag, const char* det, bool quietly)
     709 {
     710     //  (ox,cerenkov,1,dayabay)  ->   (dayabay,cerenkov,1,ox)
     711     //
     712     //     arg order twiddling done here is transitional to ease the migration 
     713     //     once working in the close to old arg order, can untwiddling all the calls
     714     //
     715     std::string path = NPYBase::path(det, source, tag, tfmt );
     716     return load(path.c_str(),quietly);
     717 }
     718 template <typename T>
     719 void NPY<T>::save(const char* tfmt, const char* source, const char* tag, const char* det)
     720 {
     721     std::string path_ = NPYBase::path(det, source, tag, tfmt );
     722     save(path_.c_str());
     723 }
     724 
     

DirectGenstep
~~~~~~~~~~~~~~~~

::

    epsilon:optickscore blyth$ opticks-find getDirectGenstep
    ./opticksgeo/OpticksGen.cc:NPY<float>* OpticksGen::getDirectGensteps() const { return m_direct_gensteps ; }
    ./cfg4/CGenerator.cc:    NPY<float>* dgs = m_gen->getDirectGensteps();
    ./g4ok/G4Opticks.cc:    const char* gspath = m_ok->getDirectGenstepPath(); 
    ./optickscore/Opticks.cc:const char* Opticks::getDirectGenstepPath() const { return m_resource->getDirectGenstepPath() ; } 
    ./optickscore/Opticks.cc:    const char* path = getDirectGenstepPath();
    ./optickscore/Opticks.cc:    std::string path = getDirectGenstepPath();
    ./boostrap/BOpticksResource.cc:const char* BOpticksResource::getDirectGenstepPath() const { return m_directgensteppath ; } 
    ./opticksgeo/OpticksGen.hh:        NPY<float>*          getDirectGensteps() const ;
    ./optickscore/Opticks.hh:       const char*          getDirectGenstepPath() const ; 
    ./boostrap/BOpticksResource.hh:       const char* getDirectGenstepPath() const ;
    epsilon:opticks blyth$ 

::

     14 const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE_NOTAG = "$OPTICKS_EVENT_BASE/evt/$1/$2" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
     15 const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE       = "$OPTICKS_EVENT_BASE/evt/$1/$2/$3" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
     16 const char* BOpticksEvent::OVERRIDE_EVENT_BASE = NULL ;
     17 
     18 const int BOpticksEvent::DEFAULT_LAYOUT_VERSION = 2 ;
     19 int BOpticksEvent::LAYOUT_VERSION = 2 ;
     20 


OPTICKS_EVENT_BASE
~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:opticks blyth$ opticks-find OPTICKS_EVENT_BASE
    ./boostrap/BFile.cc:           else if(evalue.compare("OPTICKS_EVENT_BASE")==0) 
    ./boostrap/BFile.cc:               LOG(verbose) << "expandvar replacing OPTICKS_EVENT_BASE  with " << evalue ; 
    ./boostrap/BOpticksEvent.cc:const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE_NOTAG = "$OPTICKS_EVENT_BASE/evt/$1/$2" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
    ./boostrap/BOpticksEvent.cc:const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE       = "$OPTICKS_EVENT_BASE/evt/$1/$2/$3" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
    ./boostrap/BOpticksEvent.cc:       LOG(debug) << "BOpticksEvent::directory_template OVERRIDE_EVENT_BASE replacing OPTICKS_EVENT_BASE with " << OVERRIDE_EVENT_BASE ; 
    ./boostrap/BOpticksEvent.cc:       boost::replace_first(deftmpl, "$OPTICKS_EVENT_BASE/evt", OVERRIDE_EVENT_BASE );
    ./ana/ncensus.py:    c = Census("$OPTICKS_EVENT_BASE/evt")
    ./ana/nload.py:DEFAULT_BASE = "$OPTICKS_EVENT_BASE/evt"
    ./ana/base.py:        self.setdefault("OPTICKS_EVENT_BASE",      os.path.expandvars("/tmp/$USER/opticks") )
    epsilon:opticks blyth$ 



BFile.cc OPTICKS_EVENT_BASE is not an envvar but it is internally treated a bit like one, which works
as all file access goes thru BFile::FormPath::

    087 std::string expandvar(const char* s)
     88 {
     89     fs::path p ;
     90 
     91     std::string dollar("$");
     92     boost::regex e("(\\$)(\\w+)(.*?)"); // eg $HOME/.opticks/hello
     93     boost::cmatch m ;
     94 
     95     if(boost::regex_match(s,m,e))
     96     {
     97         //dump(m);  
     98 
     99         unsigned int size = m.size();
    100 
    101         if(size == 4 && dollar.compare(m[1]) == 0)
    102         {
    103            std::string key = m[2] ;
    104 
    105            const char* evalue_ = SSys::getenvvar(key.c_str()) ;
    106 
    107            std::string evalue = evalue_ ? evalue_ : key ;
    108 
    109            if(evalue.compare("TMP")==0) //  TMP envvar not defined
    110            {
    111                evalue = usertmpdir("/tmp","opticks", NULL);
    112                LOG(verbose) << "expandvar replacing TMP with " << evalue ;
    113            }
    114            else if(evalue.compare("TMPTEST")==0)
    115            {
    116                evalue = usertmpdir("/tmp","opticks","test");
    117                LOG(verbose) << "expandvar replacing TMPTEST with " << evalue ;
    118            }
    119            else if(evalue.compare("OPTICKS_EVENT_BASE")==0)
    120            {
    121                evalue = usertmpdir("/tmp","opticks",NULL);
    122                LOG(verbose) << "expandvar replacing OPTICKS_EVENT_BASE  with " << evalue ;
    123            }
    124 
    125 
    126            p /= evalue ;
    127 
    128            std::string tail = m[3] ;
    129 
    130            p /= tail ;




Direct Route key setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~

CerenkovMinimal::

     18 void RunAction::BeginOfRunAction(const G4Run*)
     19 {
     20     LOG(info) << "." ;
     21 #ifdef WITH_OPTICKS
     22     G4VPhysicalVolume* world = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume() ;
     23     assert( world ) ;
     24     bool standardize_geant4_materials = true ;   // required for alignment 
     25     G4Opticks::GetOpticks()->setGeometry(world, standardize_geant4_materials );
     26 #endif
     27 }

Direct route, keyspec required to be set prior to Opticks instanciation::

    139 GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
    140 {
    141     const char* keyspec = X4PhysicalVolume::Key(top) ;
    142     BOpticksKey::SetKey(keyspec);
    143     LOG(error) << " SetKey " << keyspec  ;
    144 
    145     Opticks* ok = new Opticks(0,0, fEmbeddedCommandLine);  // Opticks instanciation must be after BOpticksKey::SetKey
    146 


Resource booting at Opticks instanciation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     28 BOpticksResource::BOpticksResource()
     29     :
     30     m_log(new SLog("BOpticksResource::BOpticksResource","",debug)),
     31     m_setup(false),
     32     m_key(BOpticksKey::GetKey()),   // will be NULL unless BOpticksKey::SetKey has been called 
     33     m_id(NULL),

::
 
     248 void OpticksResource::init()
     249 {
     250    LOG(LEVEL) << "OpticksResource::init" ;
     251 
     252    BStr::split(m_detector_types, "GScintillatorLib,GMaterialLib,GSurfaceLib,GBndLib,GSourceLib", ',' );
     253    BStr::split(m_resource_types, "GFlags,OpticksColors", ',' );
     254 
     255    readG4Environment();
     256    readOpticksEnvironment();
     257 
     258    if( m_key )
     259    {
     260        setupViaKey();    // from BOpticksResource base
     261    }
     262    else
     263    {
     264        readEnvironment();
     265    }
     266 
     267    readMetadata();
     268    identifyGeometry();
     269    assignDetectorName();
     270    assignDefaultMaterial();
     271 
     272    LOG(LEVEL) << "OpticksResource::init DONE" ;
     273 }




::

    In [1]: c = np_load("CerenkovMinimal/CAlignEngine.npy")

    In [2]: c
    Out[2]: array([15, 15,  9, ...,  0,  0,  0], dtype=int32)

    In [3]: c.shape
    Out[3]: (100000,)

    In [4]: c
    Out[4]: array([15, 15,  9, ...,  0,  0,  0], dtype=int32)

    In [7]: c[:230]
    Out[7]: 
    array([ 15,  15,   9,  19,   9,   9,   9,   9,  15,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,  15,  15,  15,  15,   9,  15,   9,  15,   9,   9,   9,   9,  19,   9,  15,   9,   9,   9,   9,
             9,   9,  15,   9,  15,  15,  15,   9,   9,  13,   9,  15,   9,   9,   9,   9,   9,   9,   9,  15,  46,   9,  15,  15,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,  15,  15,   9,   9,
             9,   9,  15,   9,  15,  15,   9,   9,   9,  15,   9,   9,  15,   9,   9,   9,   9,  15,  15,  15,   9,   9,  15,   9,  19,   9,  15,   9,  15,   9,  15,  22,  15,   9,  15,   9,  15,   9,
            25,  15,  15,  81,   9,   9,   9,   9,   9,   9,   9,  15,   9,   9,   9,   9,   9,  15,   9,  15,  15,   9,   9,   9,  15,  15,   9,   9,  15,  15,   9,  15,   9,   9,  15,   9,   9,  15,
             9,   9,   9,  15,   9,   9,   9,  15,  19,   9,   9,  15,   9,   9,  15,  15,   9,   9,  15,  15,  15,   9,  13,   9,   9,  15,   9,  15,  15,   9,   9,   9, 273,   9,   9,   9,  15,   9,
             9,   9,   9,  15,   9,  15,  81,   9,  35,  15,  15,   9,  15,  15,  15,   9,  15,  15,   9,  15,   9,   9,  19,   9,  15,   9,  25,   9,  15,   9,   9,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=int32)

    In [10]: c[221]
    Out[10]: 0

    In [11]: c[220]
    Out[11]: 9

    In [12]: c[184]    ## this photon should show up as discrepant, as its cursor cycled
    Out[12]: 273



