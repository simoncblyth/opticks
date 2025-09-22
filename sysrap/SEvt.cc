
#include <limits>
#include <array>
#include <csignal>

#include "scuda.h"
#include "squad.h"
#include "squadx.h"
#include "sstamp.h"
#include "sprof.h"

#include "sphoton.h"
#include "srec.h"
#include "sseq.h"
#include "ssys.h"
#include "sstate.h"
#include "stag.h"
#include "sevent.h"
#include "sctx.h"
#include "sdebug.h"
#include "stran.h"
#include "stree.h"
#include "strid.h"
#include "stimer.h"
#include "spath.h"
#include "sdirectory.h"
#include "sstr.h"
#include "ssys.h"
#include "SLOG.hh"

#include "NP.hh"
#include "NPX.h"
#include "NPFold.h"
#include "SGeo.hh"
#include "SEvt.hh"
#include "SEvent.hh"
#include "SSim.hh"
#include "SEventConfig.hh"
#include "SFrameGenstep.hh"
#include "OpticksGenstep.h"
#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"
#include "SComp.h"
#include "SProf.hh"
#include "SRecord.h"


bool SEvt::NPFOLD_VERBOSE = ssys::getenvbool(SEvt__NPFOLD_VERBOSE) ;
bool SEvt::GATHER = ssys::getenvbool(SEvt__GATHER) ;
bool SEvt::SAVE = ssys::getenvbool(SEvt__SAVE) ;
bool SEvt::INDEX = ssys::getenvbool(SEvt__INDEX) ;
bool SEvt::LIFECYCLE = ssys::getenvbool(SEvt__LIFECYCLE) ;
bool SEvt::FRAME = ssys::getenvbool(SEvt__FRAME) ;
bool SEvt::MINIMAL = ssys::getenvbool(SEvt__MINIMAL) ;
bool SEvt::MINTIME = ssys::getenvbool(SEvt__MINTIME) ;
bool SEvt::DIRECTORY = ssys::getenvbool(SEvt__DIRECTORY) ;
bool SEvt::CLEAR_SIGINT = ssys::getenvbool(SEvt__CLEAR_SIGINT) ;
bool SEvt::SIMTRACE = ssys::getenvbool(SEvt__SIMTRACE) ;
bool SEvt::EPH_ = ssys::getenvbool(SEvt__EPH) ;
bool SEvt::RUNMETA = ssys::getenvbool(SEvt__RUNMETA) ;

bool SEvt::SAVE_NOTHING = ssys::getenvbool(SEvt__SAVE_NOTHING);


const char* SEvt::descStage() const
{
    const char* st = nullptr ;
    switch(stage)
    {
        case SEvt__SEvt:         st = SEvt__SEvt_         ; break ;
        case SEvt__init:         st = SEvt__init_         ; break ;
        case SEvt__beginOfEvent: st = SEvt__beginOfEvent_ ; break ;
        case SEvt__endOfEvent:   st = SEvt__endOfEvent_   ; break ;
        case SEvt__gather:       st = SEvt__gather_       ; break ;
        default:                 st = SEvt__OTHER_        ; break ;
    }
    return st ;
}

bool SEvt::IsDefined(unsigned val){ return val != UNDEF ; }

stimer* SEvt::TIMER = new stimer ;
void SEvt::TimerStart(){ TIMER->start(); }
double SEvt::TimerDone(){ return TIMER->done() ; }
uint64_t SEvt::TimerStartCount(){ return TIMER->start_count() ; }
std::string SEvt::TimerDesc(){ return TIMER->desc() ; }


/**
SEvt::Init_RUN_META
---------------------

As this is a static it happens just as libSysRap is loaded,
very soon after starting the executable.

**/


NP* SEvt::Init_RUN_META() // static
{
    NP* run_meta = NP::Make<float>(1);
    run_meta->set_meta<std::string>("SEvt__Init_RUN_META", sprof::Now() );
    return run_meta ;
}

NP* SEvt::RUN_META = Init_RUN_META() ;

std::string* SEvt::RunMetaString() // static
{
    return RUN_META ? &(RUN_META->meta) : nullptr ;
}


NP* SEvt::UU = nullptr ;
NP* SEvt::UU_BURN = nullptr ; // NP::Load("$SEvt__UU_BURN") ;

const plog::Severity SEvt::LEVEL = SLOG::EnvLevel("SEvt", "DEBUG");
const int SEvt::GIDX = ssys::getenvint("GIDX",-1) ;
const int SEvt::PIDX = ssys::getenvint("PIDX",-1) ;
const int SEvt::MISSING_INDEX = std::numeric_limits<int>::max() ;
const int SEvt::MISSING_INSTANCE = std::numeric_limits<int>::max() ;
const int SEvt::DEBUG_CLEAR = ssys::getenvint("SEvt__DEBUG_CLEAR",-1) ;



std::array<SEvt*, SEvt::MAX_INSTANCE> SEvt::INSTANCES = {{ nullptr, nullptr }} ;

std::string SEvt::DescINSTANCE()  // static
{
    std::stringstream ss ;

    ss << "SEvt::DescINSTANCE"
       << " Count() " << Count()
       << std::endl
       << " Exists(0) " << ( Exists(0) ? "YES" : "NO " )
       << std::endl
       << " Exists(1) " << ( Exists(1) ? "YES" : "NO " )
       << std::endl
       ;
    std::string str = ss.str();
    return str ;
}



void SEvt::setStage(int stage_)
{
    stage = stage_ ;
}
int SEvt::getStage() const
{
    return stage ;
}


/**
SEvt::SEvt
-----------

Instanciation invokes SEventConfig::Initialize()

The config used depends on:

1. envvars such as OPTICKS_EVENT_MODE that can change default config values
2. static SEventConfig method calls done before SEvt instanciation
   that change the default config values
3. available VRAM as detected by (scontext)SEventConfig::CONTEXT

**/

SEvt::SEvt()
    :
    cfgrc(SEventConfig::Initialize()),
    index(MISSING_INDEX),
    instance(MISSING_INSTANCE),
    stage(SEvt__SEvt),
    gather_metadata_notopfold(0),
    t_BeginOfEvent(0),
#ifndef PRODUCTION
    t_setGenstep_0(0),
    t_setGenstep_1(0),
    t_setGenstep_2(0),
    t_setGenstep_3(0),
    t_setGenstep_4(0),
    t_setGenstep_5(0),
    t_setGenstep_6(0),
    t_setGenstep_7(0),
    t_setGenstep_8(0),
#endif
    t_PreLaunch(0),
    t_PostLaunch(0),
    t_EndOfEvent(0),
    t_PenultimatePoint(0),
    t_LastPoint(0),
    t_Launch(-2.),
    selector(new sphoton_selector(SEventConfig::HitMask())),
    evt(new sevent),
    dbg(new sdebug),
    input_genstep(nullptr),
    input_photon(nullptr),
    input_photon_transformed(nullptr),
    g4state(nullptr),
    random(nullptr),
    random_array(nullptr),
    provider(this),   // overridden with SEvt::setCompProvider for device running from QEvent::init
    topfold(new NPFold),
    fold(nullptr),
    extrafold(new NPFold),
    cf(nullptr),
    hostside_running_resize_done(false),
    gather_done(false),
    is_loaded(false),
    is_loadfail(false),
    numgenstep_collected(0u),   // updated by addGenstep
    numphoton_collected(0u),   // updated by addGenstep
    numphoton_genstep_max(0u),
    clear_genstep_vector_count(0),
    clear_output_vector_count(0),
    gather_total(0),
    genstep_total(0),
    photon_total(0),
    hit_total(0),
    addGenstep_array(0)
{
    init();
}

/**
SEvt::init
-----------

Only configures array maxima, no allocation yet.
Device side allocation happens in QEvent::setGenstep QEvent::setNumPhoton

Initially SEvt is set as its own SCompProvider,
allowing U4RecorderTest/SEvt::save to gather the component
arrays provided from SEvt.

For device running the SCompProvider is overridden to
become QEvent allowing SEvt::save to persist the
components gatherered from device buffers.

**/

void SEvt::init()
{
    if(NPFOLD_VERBOSE) topfold->set_verbose();

    setStage(SEvt__init);

    LOG_IF(info, LIFECYCLE) << id() ;

    LOG(LEVEL) << "[" ;
    //INSTANCE = this ;    // no-longer automated, rely on static creation methods to set INSTANCES[0] or [1]

    evt->init();    // array maxima set according to SEventConfig values
    dbg->zero();

    LOG(LEVEL) << evt->desc() ; // mostly zeros at this juncture

    SEventConfig::GatherCompList(gather_comp);  // populate gather_comp vector based on GatherCompMask
    SEventConfig::SaveCompList(save_comp);      // populate save_comp vector based on SaveCompMask


    LOG(LEVEL) << " SEventConfig::DescGatherComp "  << SEventConfig::DescGatherComp() ;
    LOG(LEVEL) << " SEventConfig::DescSaveComp "    << SEventConfig::DescSaveComp() ;
    LOG(LEVEL) << descComp() ;

    //initInputGenstep();   // for per-event genstep moved to SEvt::beginOfEvent/SEvt::addInputGenstep



    initInputPhoton();

    initG4State();        // HMM: G4State not an every-event thing ? first event only ?
    LOG(LEVEL) << "]" ;
}

void SEvt::setFoldVerbose(bool v)
{
    topfold->set_verbose(v);
    if(fold) fold->set_verbose(v);
}


const char* SEvt::GetSaveDir(int idx) // static
{
    return Exists(idx) ? Get(idx)->getSaveDir() : nullptr ;
}
const char* SEvt::getSaveDir() const { return topfold->savedir ; }
const char* SEvt::getLoadDir() const { return topfold->loaddir ; }
int SEvt::getTotalItems() const { return topfold->total_items() ; }

/**
SEvt::getSearchCFbase
----------------------

Search for CFBase geometry folder corresponding to event arrays based on
the loaded/saved SEvt directories eg with::

    spath::SearchDirUpTreeWithFile(dir, "CSGFoundry/solid.npy")

As the start dirs tried are loaddir and savedir this has no possibility of working prior
to a load or save having been done. As this is typically used for loaded SEvt
this is not much of a limitation.

For this to succeed to find the geometry requires event arrays are saved
into folders with suitable proximity to the corresponding geometry folders.

For example use this to load an event and associate it with a geometry,
allowing dumping or access to the photons in any frame::

    SEvt* sev = SEvt::Load() ;
    const char* cfbase = sev->getSearchCFBase() ;
    const CSGFoundry* fd = CSGFoundry::Load(cfbase);
    sev->setGeo(fd);
    sev->setFrame(39216);
    std::cout << sev->descFull() ;

See CSG/tests/CSGFoundry_SGeo_SEvt_Test.sh

**/

const char* SEvt::getSearchCFBase() const
{
    const char* loaddir = getLoadDir();
    const char* savedir = getSaveDir();
    const char* cfbase = nullptr ;
    if(cfbase == nullptr && loaddir) cfbase = spath::SearchDirUpTreeWithFile(loaddir, SearchCFBase_RELF );
    if(cfbase == nullptr && savedir) cfbase = spath::SearchDirUpTreeWithFile(savedir, SearchCFBase_RELF) ;
    return cfbase ;
}


const char* SEvt::INPUT_GENSTEP_DIR = spath::Resolve("${SEvt__INPUT_GENSTEP_DIR:-$HOME/.opticks/InputGensteps}") ;
const char* SEvt::INPUT_PHOTON_DIR = spath::Resolve("${SEvt__INPUT_PHOTON_DIR:-$HOME/.opticks/InputPhotons}") ;


/**
SEvt::ResolveInputArray
------------------------

When *spec* starts with a letter AZaz it is assumed to be a relative path within
the SEvt__INPUT_GENSTEP_DIR which defaults to $HOME/.opticks/InputGensteps.

When *spec* starts with a non-letter such as "$" or "/" it is assumed
to specify an absolute path and the *dir* argument is ignored.

**/

const char* SEvt::ResolveInputArray(const char* spec, const char* dir) // static
{
    assert(strlen(spec) > 0 && sstr::EndsWith(spec, ".npy") );
    const char* path = sstr::StartsWithLetterAZaz(spec) ?  spath::Resolve(dir, spec) : spath::Resolve( spec ) ;
    return path ;
}

NP* SEvt::LoadInputArray(const char* path) // static
{
    NP* a = NP::Load(path);
    LOG_IF(fatal, a == nullptr) << " FAILED to load input array from path " << path ;

    LOG(LEVEL)
        << " path " << ( path ? path : "-" )
        << " a.sstr " << ( a ? a->sstr() : "-" )
        ;

    assert( a ) ;
    assert( a->has_shape(-1,4,4) || a->has_shape(-1,6,4));
    assert( a->shape[0] > 0 );

    return a ;
}



/**
SEvt::LoadInputGenstep_array
-----------------------------

::

    export OPTICKS_INPUT_GENSTEP=$BASE/jok-tds/ALL0/p001/genstep.npy
    ##export OPTICKS_INPUT_GENSTEP=$BASE/jok-tds/ALL0


HMM: when using a sequence of InputGenstep files need to
do this for every event, invoking with an idx argument ..
so this cannot be static

**/


NP* SEvt::LoadInputGenstep_array(int idx) // static
{
    const char* spec = SEventConfig::InputGenstep(idx);
    return spec ? LoadInputGenstep_array(spec) : nullptr ;
}
NP* SEvt::LoadInputGenstep_array(const char* spec) // static
{
    const char* path = ResolveInputArray( spec, INPUT_GENSTEP_DIR );
    NP* a = LoadInputArray(path);
    assert( a->has_shape(-1,6,4) );
    return a ;
}


/**
SEvt::LoadInputPhoton
----------------------

This is invoked by SEvt::initInputPhoton which is invoked by SEvt::init at instanciation.

Resolving the input string to a path is done in one of two ways:

1. if the string starts with a letter A-Za-z eg "inphoton.npy" or "RandomSpherical10.npy"
   it is assumed to be the name of a .npy file within the default SEvt__INPUT_PHOTON_DIR
   of $HOME/.opticks/InputPhotons.

   Create such files with ana/input_photons.py

2. if the string does not start with a letter eg /path/to/some/dir/file.npy or $TOKEN/path/to/file.npy
   it is passed unchanged to  spath::Resolve

**/

NP* SEvt::LoadInputPhoton() // static
{
    const char* spec = SEventConfig::InputPhoton();
    return spec ? LoadInputPhoton(spec) : nullptr ;
}
NP* SEvt::LoadInputPhoton(const char* spec)
{
    bool is_ipr = IsInputPhotonRecord(spec);
    NP* a = nullptr ;
    if(!is_ipr)
    {
        a = LoadInputPhoton_photon(spec);
    }
    else
    {
        a = LoadInputPhoton_record(spec);
    }
    return a ;
}

bool SEvt::IsInputPhotonRecord( const char* spec )
{
    const char* path = spath::Resolve(spec);
    bool is_record = sstr::EndsWith(path, "record.npy" );
    return is_record ;
}

NP* SEvt::LoadInputPhoton_photon(const char* spec)
{
    bool is_simtrace = !SFrameGenstep::HasConfigEnv() && SEventConfig::IsRGModeSimtrace() ;
    LOG_IF(fatal, is_simtrace)
        << " simtrace with ordinary input photons is not supported "
        ;
    assert(!is_simtrace);
    if(is_simtrace) std::raise(SIGINT);

    const char* path = ResolveInputArray( spec, INPUT_PHOTON_DIR );
    NP* a = LoadInputArray(path);
    assert( a->has_shape(-1,4,4) );

    float t0 = SEventConfig::InputPhotonChangeTime();
    bool change_time = t0 >= 0.f ;

    LOG_IF(fatal, change_time)
        << " SEventConfig::InputPhotonChangeTime " << t0
        << " change_time " << ( change_time ? "YES" : "NO " )
        ;

    if(change_time) sphoton::ChangeTimeInsitu(a, t0);
    return a ;
}

/**
SEvt::LoadInputPhoton_record
------------------------------

This is is invoked when OPTICKS_INPUT_PHOTON
resolves to a path ending with record.npy

Creates input photon array by interpolation of
step point positions from record array at
simulation time given by OPTICKS_INPUT_PHOTON_RECORD_TIME
(in ns).

**/


NP* SEvt::LoadInputPhoton_record(const char* spec)
{
    bool is_simtrace = !SFrameGenstep::HasConfigEnv() && SEventConfig::IsRGModeSimtrace() ;

    const char* spec2 = SEventConfig::InputPhoton();
    const char* slice = SEventConfig::InputPhotonRecordSlice();
    const char* iprt = SEventConfig::InputPhotonRecordTime();

    assert( spec && spec2 && strcmp( spec2, spec) == 0 );

    const char* path = spath::Resolve(spec);
    bool is_record = sstr::EndsWith(path, "record.npy" );
    assert( is_record );

    const char* fold = spath::Dirname(path);

    SRecord* sr = SRecord::Load(fold, slice );
    NP* a = is_simtrace ? sr->getSimtraceAtTime(iprt) : sr->getPhotonAtTime(iprt) ;

    std::cout
        << "SEvt::LoadInputPhoton_record"
        << " is_simtrace " << ( is_simtrace ? "YES" : "NO " )
        << " spec [" << ( spec ? spec : "-" ) << "]\n"
        << " path [" << ( path ? path : "-" ) << "]\n"
        << " fold [" << ( fold ? fold : "-" ) << "]\n"
        << " slice [" << ( slice ? slice : "-" ) << "]\n"
        << " iprt [" << ( iprt ? iprt : "-" ) << "]\n"
        << " is_record " << ( is_record ? "YES" : "NO " ) << "\n"
        << " a " << ( a ? a->sstr() : "-" ) << "\n"
        ;

    return a ;
}



/**
SEvt::initInputGenstep_array
-----------------------------

Invoked from SEvt::beginOfEvent after SEvt::setIndex
Formerly invoked once only from SEvt::init

**/

void SEvt::initInputGenstep_array()
{
    NP* ig = LoadInputGenstep_array(index) ;
    setInputGenstep_array(ig);
}
void SEvt::setInputGenstep_array(NP* ig)
{
    if(ig == nullptr) return ;
    input_genstep = ig ;
    assert( input_genstep->has_shape(-1,6,4) );
    int numgenstep = input_genstep->shape[0] ;
    bool numgenstep_expect = numgenstep > 0 ;
    if(!numgenstep_expect) std::raise(SIGINT) ;
    assert( numgenstep_expect );
}


NP* SEvt::getInputGenstep_array() const { return input_genstep ; }
bool SEvt::hasInputGenstep_array() const { return input_genstep != nullptr ; }
bool SEvt::hasInputGenstep_arrayPath() const { return SEventConfig::InputGenstepPathExists(index) ; }



/**
SEvt::initInputPhoton
-----------------------

This is invoked by SEvt::init on instanciating the SEvt instance
The default "SEventConfig::InputPhoton()" is nullptr meaning no input photons.
This can be changed by setting an envvar in the script that runs the executable, eg::

   export OPTICKS_INPUT_PHOTON=CubeCorners.npy
   export OPTICKS_INPUT_PHOTON=$HOME/somedir/path/to/inphoton.npy

Or within the code of the executable, typically in the main prior to SEvt instanciation,
using eg::

   SEventConfig::SetInputPhoton("CubeCorners.npy")
   SEventConfig::SetInputPhoton("$HOME/somedir/path/to/inphoton.npy")

When non-null it is resolved into a path and the array loaded at SEvt instanciation
by SEvt::LoadInputPhoton

**/

void SEvt::initInputPhoton()
{
    bool is_CEGS_simtrace = SFrameGenstep::HasConfigEnv() && SEventConfig::IsRGModeSimtrace() ;
    if(is_CEGS_simtrace) return ;  // skip input photon setup when in simtrace mode with CEGS envvar

    NP* ip = LoadInputPhoton() ;
    setInputPhoton(ip);
}


void SEvt::setInputPhoton(NP* p)
{
    if(p == nullptr) return ;
    input_photon = p ;
    bool input_photon_expect = input_photon->has_shape(-1,4,4) ;
    if(!input_photon_expect) std::raise(SIGINT) ;
    assert( input_photon_expect );

    int numphoton = input_photon->shape[0] ;
    bool numphoton_expect = numphoton > 0 ;
    if(!numphoton_expect) std::raise(SIGINT) ;
    assert( numphoton_expect  );
}






/**
SEvt::getInputPhoton_
----------------------

This variant always provides the untransformed input photons.
That will be nullptr unless OPTICKS_INPUT_PHOTON is defined.

**/
NP* SEvt::getInputPhoton_() const { return input_photon ; }
bool SEvt::hasInputPhoton() const { return input_photon != nullptr ; }


/**
SEvt::getInputPhoton
---------------------

Returns the transformed input photon if present.
For the transformed photons to  be present it is necessary to have called SEvt::setFrame
That is done from on high by G4CXOpticks::setupFrame which gets invoked by G4CXOpticks::setGeometry

The frame and corresponding transform used can be controlled by several envvars,
see CSGFoundry::getFrameE. Possible envvars include:

+------------------------------+----------------------------+
| envvar                       | Examples                   |
+==============================+============================+
| INST                         |                            |
+------------------------------+----------------------------+
| MOI                          | Hama:0:1000 NNVT:0:1000    |
+------------------------------+----------------------------+
| OPTICKS_INPUT_PHOTON_FRAME   |                            |
+------------------------------+----------------------------+


**/
NP* SEvt::getInputPhoton() const {  return input_photon_transformed ? input_photon_transformed : input_photon  ; }
bool SEvt::hasInputPhotonTransformed() const { return input_photon_transformed != nullptr ; }



NP* SEvt::gatherInputGenstep() const
{
    NP* ig = getInputGenstep_array();
    NP* ign = nullptr ;
    if(ig)
    {
        ign = ig->ebyte == 8 ? NP::MakeNarrow(ig) : ig->copy() ;
    }
    return ign ;
}



/**
SEvt::gatherInputPhoton
-------------------------

To avoid issues with inphoton and saving 2nd events,
treat the inphoton more like other arrays by having a distinct
narrowed inphoton copy for each event.

HMM: unlike other gathered component this is not being
added to the fold, so will not be cleared ?

TODO : switch to narrowing at initialization, so can avoid the gather


**/

NP* SEvt::gatherInputPhoton() const
{
    NP* ip = getInputPhoton();
    NP* ipn = nullptr ;
    if(ip)
    {
        ipn = ip->ebyte == 8 ? NP::MakeNarrow(ip) : ip->copy() ;
    }
    return ipn ;
}


/**
SEvt::initG4State
-------------------

Called by SEvt::init but does nothing unless SEventConfig::IsRunningModeG4StateSave()

HMM: is this the right place ? It depends on the RunningMode.

**/

void SEvt::initG4State()
{
    if(SEventConfig::IsRunningModeG4StateSave())
    {
        LOG(LEVEL) << "SEventConfig::IsRunningModeG4StateSave creating g4state array " ;
        NP* state = makeG4State();
        setG4State(state);
    }
}

/**
SEvt::makeG4State
---------------------

Not invoked by default.

See U4Recorder::PreUserTrackingAction_Optical

* HMM: thats a bit spagetti, config control ?

Item values of 2*17+4=38 is appropriate for the default Geant4 10.4.2 random engine: MixMaxRng
See::

     g4-cls MixMaxRng
     g4-cls RandomEngine
     g4-cls Randomize


TODO: half the size with "unsigned" not "unsigned long",
to workaround wasteful CLHEP API which uses "unsigned long" but only needs 32 bit elements

**/

NP* SEvt::makeG4State() const
{
     const char* spec = SEventConfig::G4StateSpec() ; // default spec 1000:38

     std::vector<int> elem ;
     sstr::split<int>(elem, spec, ':');
     assert( elem.size() == 2 );

     int max_states = elem[0] ;
     int item_values = elem[1] ;

     NP* a =  NP::Make<unsigned long>(max_states, item_values );


     LOG(info)
         << " SEventConfig::G4StateSpec() " << spec
         << " max_states " << max_states
         << " item_values " << item_values
         << " a.sstr " << a->sstr()
         ;
     return a ;
}

void SEvt::setG4State( NP* state) { g4state = state ; }
NP* SEvt::gatherG4State() const {  return g4state ; } // gather is used prior to persisting, get is used after loading
const NP* SEvt::getG4State() const {  return topfold->get(SComp::Name(SCOMP_G4STATE)) ; }


/**
SEvt::setFrame
------------------

As it is necessary to have the geometry to provide the frame this
is now split from eg initInputPhotons.

**simtrace running**
    MakeCenterExtentGensteps based on the given frame.

**simulate inputphoton running**
    MakeInputPhotonGenstep and m2w (model-2-world)
    transforms the photons using the frame transform

Formerly(?) for simtrace and input photon running with or without a transform
it was necessary to call this for every event due to the former call to addInputGenstep,
but now that the genstep setup is moved to SEvt::beginOfEvent it is only needed
to call this for each frame, usually once only.

**/


void SEvt::setFrame(const sframe& fr )
{
    const char* name = fr.get_name() ;
    LOG_IF(info, FRAME)
        << " [" << SEvt__FRAME << "]"
        << " fr.get_name " << ( name ? name : "-" ) << "\n"
        << " fr.desc\n"
        << fr.desc()
        ;
    frame = fr ;
    transformInputPhoton();
}

void SEvt::setFramePlaceholder()
{
    sframe fr = sframe::Fabricate(0.f,0.f,0.f);
    setFrame(fr);
}



const bool SEvt::transformInputPhoton_VERBOSE = ssys::getenvbool("SEvt__transformInputPhoton_VERBOSE") ;
const bool SEvt::transformInputPhoton_WIDE = ssys::getenvbool("SEvt__transformInputPhoton_WIDE") ;

/**
SEvt::transformInputPhoton
---------------------------

**/

void SEvt::transformInputPhoton()
{
    bool proceed = SEventConfig::IsRGModeSimulate() && hasInputPhoton() ;

    LOG(LEVEL) << " proceed " << ( proceed ? "YES" : "NO " ) ;

    LOG_IF(info, transformInputPhoton_VERBOSE )
        << " SEvt__transformInputPhoton_VERBOSE "
        << " SEventConfig::IsRGModeSimulate " << SEventConfig::IsRGModeSimulate()
        << " hasInputPhoton " <<  hasInputPhoton()
        << " proceed " << ( proceed ? "YES" : "NO " )
        << "\n"
        << frame.desc()
        << "\n"
        ;

    if(!proceed) return ;

    bool normalize = true ;  // normalize mom and pol after doing the transform

    NP* ipt = frame.transform_photon_m2w( input_photon, normalize );

    if(transformInputPhoton_WIDE)  // see notes/issues/G4ParticleChange_CheckIt_warnings.rst
    {
        input_photon_transformed = ipt ;
    }
    else
    {
        input_photon_transformed = ipt->ebyte == 8 ? NP::MakeNarrow(ipt) : ipt ;
        // narrow here to prevent immediate A:B difference with Geant4 seeing double precision
        // and Opticks float precision
    }
}





NP* SEvt::createInputGenstep_simtrace()
{
    NP* igs = nullptr ;

    assert(SEventConfig::IsRGModeSimtrace());
    bool with_CEGS = SFrameGenstep::HasConfigEnv();
    bool with_ip = hasInputPhoton();
    LOG_IF(info, SIMTRACE)
        << "[" << SEvt__SIMTRACE << "] "
        << " with_CEGS " << ( with_CEGS ? "YES" : "NO " )
        << " with_ip " << ( with_ip ? "YES" : "NO " )
        ;


    if(with_CEGS)
    {
        const char* frs = frame.get_frs() ; // nullptr when default -1 : meaning all geometry
        LOG_IF(info, SIMTRACE )
            << "[" << SEvt__SIMTRACE << "] "
            << " frame.get_frs " << ( frs ? frs : "-" ) ;
            ;

        //if(frs) SEventConfig::SetEventReldir(frs); // dont do that, default is more standard
        // doing this is hangover from separate simtracing of related volumes presumably

        igs = SFrameGenstep::MakeCenterExtentGenstep_FromFrame(frame);
        LOG_IF(info, SIMTRACE)
            << "[" << SEvt__SIMTRACE << "] "
            << " simtrace igs " << ( igs ? igs->sstr() : "-" )
            ;

    }
    else
    {
        // simtrace mode without CEGS and with OPTICKS_INPUT_PHOTON that must be via record.npy SRecord::getSimtraceAtTime
        if( with_ip )
        {
            igs = SEvent::MakeInputPhotonGenstep(input_photon, OpticksGenstep_INPUT_PHOTON_SIMTRACE, nullptr) ;
            assert(igs);
        }
        else
        {
            // low level CSG/CSGSimtrace does this
        }
    }
    return igs ;
}


/**
SEvt::createInputGenstep_simulate
--------------------------------

This branches between ECPU and EGPU because
U4VPrimaryGenerator::GeneratePrimaries needs
torch gensteps before U4Recorder::BeginOfEventAction
So have to skip the genstep setup for ECPU as it should
have been done already.

**/



NP* SEvt::createInputGenstep_simulate()
{
    NP* igs = nullptr ;

    bool has_torch = SEventConfig::IsRunningModeTorch() ; // TODO: rename to InputTorch
    int inputs = int(hasInputGenstep_arrayPath()) + int(hasInputPhoton()) + int(has_torch) ;

    LOG_IF(info, LIFECYCLE) << " inputs " << inputs  ;
    LOG_IF(fatal, inputs > 1 )
        << " CANNOT COMBINE INPUTS : input_photon/input_genstep/torch "
        << " index " << index
        ;
    assert( inputs == 0 || inputs == 1);
    if( inputs == 1 )
    {
        assertZeroGenstep();
        if( hasInputGenstep_arrayPath() )
        {
            initInputGenstep_array();
            igs = getInputGenstep_array() ;
        }
        else if( hasInputPhoton())
        {
            igs = SEvent::MakeInputPhotonGenstep(input_photon, OpticksGenstep_INPUT_PHOTON, &frame) ;
        }
        else if( has_torch )
        {
            if( SEvent::HasGENSTEP() )
            {
                // expected with G4CXApp.h U4Recorder running : see G4CXApp::GeneratePrimaries
                // this is because the gensteps are needed really early with Geant4 running
                igs = SEvent::GetGENSTEP() ;
            }
            else
            {
                int index_arg = getIndexArg();
                igs = SEvent::MakeTorchGenstep(index_arg);  // pass index to allow changing num photons per event
            }
        }
    }
    return igs ;
}



/**
SEvt::createInputGenstep_configured
----------------------------------

**/


NP* SEvt::createInputGenstep_configured()
{
    NP* igs = nullptr ;
    if(SEventConfig::IsRGModeSimtrace())
    {
        igs = createInputGenstep_simtrace();
    }
    else if(SEventConfig::IsRGModeSimulate())
    {
        igs = createInputGenstep_simulate();
    }
    return igs ;
}




/**
SEvt::addInputGenstep  (formerly addFrameGenstep)
--------------------------------------------------

This is invoked from SEvt::beginOfEvent after invokation of SEvt::setIndex

For hostside simtrace and input photon running this must be called
at the start of every event cycle to add the gensteps which trigger
the allocations for result vectors.


**/

void SEvt::addInputGenstep()
{
    LOG_IF(info, LIFECYCLE) << id() ;
    LOG(LEVEL);

    NP* igs = createInputGenstep_configured();
    addGenstep(igs);
}

void SEvt::assertZeroGenstep()
{
    int prior_genstep = genstep.size() ;
    bool prior_genstep_zero = prior_genstep == 0 ;

    LOG_IF(fatal, !prior_genstep_zero )
        << " FIND genstep WHEN NONE ARE EXPECTED  "
        << " index " << index
        << " instance " << instance
        << " isECPU " << isECPU()
        << " isEGPU " << isEGPU()
        << " prior_genstep_zero " << ( prior_genstep_zero ? "YES" : "NO " )
        << " prior_genstep " << prior_genstep
        ;
    assert( prior_genstep_zero ) ;
}



const char* SEvt::getFrameId()    const { return frame.getFrameId() ; }
const NP*   SEvt::getFrameArray() const { return frame.getFrameArray() ; }

const char* SEvt::GetFrameId(int idx){    return Exists(idx) ? Get(idx)->getFrameId() : nullptr ; }
const NP*   SEvt::GetFrameArray(int idx){ return Exists(idx) ? Get(idx)->getFrameArray() : nullptr ; }


/**
SEvt::setFrame_HostsideSimtrace
---------------------------------

Called from SEvt::setFrame when sframe::is_hostside_simtrace, eg at behest of U4Simtrace

**/

void SEvt::setFrame_HostsideSimtrace()
{
    unsigned num_photon_gs = getNumPhotonFromGenstep();
    unsigned num_photon_evt = evt->num_photon ;
    LOG_IF(info, SIMTRACE)
         << "frame.is_hostside_simtrace"
         << " num_photon_gs " << num_photon_gs
         << " num_photon_evt " << num_photon_evt
         ;

    assert( num_photon_gs == num_photon_evt );
    setNumSimtrace( num_photon_gs );

    LOG_IF(info, SIMTRACE) << " before hostside_running_resize simtrace.size " << simtrace.size() ;

    hostside_running_resize();

    LOG_IF(info, SIMTRACE) << " after hostside_running_resize simtrace.size " << simtrace.size() ;

    SFrameGenstep::GenerateSimtracePhotons( simtrace, genstep );
}



/**
SEvt::setGeo
-------------

SGeo is a protocol for geometry access fulfilled by CSGFoundry (and formerly by GGeo)

Canonical invokation is from G4CXOpticks::setGeometry
This connection between the SGeo geometry and SEvt is what allows
the appropriate instance frame to be accessed. That is vital for
looking up the sensor_identifier and sensor_index.

TODO: replace this with stree.h based approach

**/

void SEvt::setGeo(const SGeo* cf_)
{
    cf = cf_ ;
    tree = cf->getTree();
}

/**
SEvt::setFrame
-----------------

This method asserts that SEvt::setGeo has been called
as that is needed to provide access to sframe via
the SGeo protocol base of either GGeo OR CSGFoundry

TODO: replace this with stree.h based approach

**/
void SEvt::setFrame(unsigned ins_idx)
{
    LOG_IF(fatal, cf == nullptr) << "must SEvt::setGeo before being can access frames " ;
    assert(cf);
    sframe fr ;
    int rc = cf->getFrame(fr, ins_idx) ;
    if(rc!=0) std::raise(SIGINT);
    assert( rc == 0 );
    fr.prepare();

    setFrame(fr);
}


//// below impl order matches decl order : KEEP IT THAT WAY


/**
SEvt::CreateSimtraceEvent
---------------------------

Experimental creation of a new SEvt
that adopts the sframe of the prior SEvt
and is configured for hostside Simtrace running.

The appropriate place to use this method from
is EndOfRunAction after standard (non-simtrace)
usage of SEvt such as saving simulation SEvt
is completed.

**/

SEvt* SEvt::CreateSimtraceEvent()  // static
{
    LOG_IF(info, SIMTRACE) << "[" ;

    SEvt* prior = SEvt::Get(0);
    assert(prior);
    if(prior == nullptr ) return nullptr ;

    sframe& pfr = prior->frame ;
    pfr.set_hostside_simtrace();

    if( pfr.ce.w == 0.f )
    {
        pfr.ce.w = 200.f ;
        LOG_IF(info, SIMTRACE)
            << " kludging frame extent, this happens with U4SimulateTest "
            << " as the CSGFoundry geometry is not available causing the "
            << " SEvt sframe to be default "
            ;
    }

    // set_hostside_simtrace into frame which
    // influences the action of SEvt::setFrame
    // especially SEvt::setFrame_HostsideSimtrace which
    // generates simtrace photons

    SEventConfig::SetRGModeSimtrace(); // now added handling to redo Initialize_Comp when changing mode

    LOG_IF(info, SIMTRACE) << " SWITCH : SEventConfig::SetRGModeSimtrace " ;
    SEvt* ste = new SEvt ;
    ste->setFrame(pfr);

    LOG_IF(info, SIMTRACE) << "] ste.simtrace.size " << ste->simtrace.size() ;

    return ste ;
}



/**
SEvt::setCompProvider
----------------------

This is called for device side running only from QEvent::init

**/

void SEvt::setCompProvider(const SCompProvider* provider_)
{
    provider = provider_ ;
    LOG(LEVEL) << descProvider() ;
}

bool SEvt::isSelfProvider() const {   return provider == this ; }

std::string SEvt::descProvider() const
{
    bool is_self_provider = isSelfProvider() ;
    std::stringstream ss ;
    ss << "SEvt::descProvider"
       << " provider.getTypeName " << provider->getTypeName()
       << " that address is: " << ( is_self_provider ? "SELF" : "another object" )
       ;
    std::string s = ss.str();
    return s ;
}

/**
SEvt::gatherDomain
--------------------

Create (2,4,4) NP array and populate with quad4 from::

    sevent::get_domain
    sevent::get_config


NB an alternative route for metadata is the
SEvt::meta OR QEvent::meta that gets adopted as
the NPFold meta in SEvt::gather_components

DOMAIN IS INCREASINGLY IRRELEVANT WHEN NOT USING REC

TODO: ELIMINATE

**/

NP* SEvt::gatherDomain() const
{
    quad4 dom[2] ;
    evt->get_domain(dom[0]);
    evt->get_config(dom[1]);  // maxima, counts
    NP* domain = NP::Make<float>( 2, 4, 4 );
    domain->read2<float>( (float*)&dom[0] );
    return domain ;
}

int SEvt::Count()  // static
{
    int count = 0 ;
    if(Exists(0)) count += 1 ;
    if(Exists(1)) count += 1 ;
    return count ;
}


SEvt* SEvt::Get_EGPU(){ return SEvt::Get(EGPU) ; }
SEvt* SEvt::Get_ECPU(){ return SEvt::Get(ECPU) ; }

SEvt* SEvt::Get(int idx)  // static
{
    assert( idx == 0 || idx == 1 );
    return INSTANCES[idx] ;
}
void SEvt::Set(int idx, SEvt* inst) // static
{
    assert( idx == 0 || idx == 1 );
    INSTANCES[idx] = inst ;
    LOG(LEVEL) << " idx " << idx  << " " << DescINSTANCE()  ;
}

/**
SEvt::Create
-------------

Q: Does all SEvt creation use this ?


**/

SEvt* SEvt::Create_EGPU(){ return Create(EGPU) ; }
SEvt* SEvt::Create_ECPU(){ return Create(ECPU) ; }

SEvt* SEvt::Create(int ins)  // static
{
    //const char* alldir = spath::Resolve("ALL${VERSION:-0}") ;
    //SEvt::SetReldir(alldir);

    assert( ins == 0 || ins == 1);
    SEvt* ev = new SEvt ;
    ev->setInstance(ins) ;
    INSTANCES[ins] = ev  ;
    assert( Get(ins) == ev );
    LOG(LEVEL) << " ins " << ins  << " " << DescINSTANCE()  ;
    return ev  ;
}

bool SEvt::isEGPU() const { return instance == EGPU ; }
bool SEvt::isECPU() const { return instance == ECPU ; }

/**
SEvt::isFirstEvtInstance SEvt::isLastEvtInstance
--------------------------------------------------

When have both ECPU and EGPU the ECPU is first
otherwise when there is only one event
whatever it is is inevitably first(and last).

**/
bool SEvt::isFirstEvtInstance() const
{
    return Exists_EGPU() && Exists_ECPU() ? isECPU() : true ;
}
bool SEvt::isLastEvtInstance() const
{
    return Exists_EGPU() && Exists_ECPU() ? isEGPU() : true ;
}




SEvt* SEvt::getSibling() const
{
    SEvt* sibling = nullptr ;
    switch(instance)
    {
        case ECPU:  sibling = Get(EGPU) ; break ;
        case EGPU:  sibling = Get(ECPU) ; break ;
        default:    sibling = nullptr ;
    }
    return sibling ;
}


bool SEvt::Exists(int idx)  // static
{
    return Get(idx) != nullptr ;
}

bool SEvt::Exists_ECPU(){ return Exists(ECPU) ; }
bool SEvt::Exists_EGPU(){ return Exists(EGPU) ; }




SEvt* SEvt::CreateOrReuse(int idx)
{
    SEvt* sev = Exists(idx) ? Get(idx) : Create(idx) ;
    LOG(LEVEL) << " idx " << idx  << " " << DescINSTANCE()  ;
    return sev ;
}



SEvt* SEvt::CreateOrReuse_EGPU(){  return CreateOrReuse(EGPU) ; }
SEvt* SEvt::CreateOrReuse_ECPU(){  return CreateOrReuse(ECPU) ; }

/**
SEvt::CreateOrReuse
---------------------

Creates 0, 1 OR 2 SEvt depending on SEventConfig::IntegrationMode()::

    OPTICKS_INTEGRATION_MODE (aka opticksMode)

+-----------------+----------+----------+--------------------------------------------+
|  opticksMode    | num SEvt | SEvt idx | notes                                      |
+=================+==========+==========+============================================+
|             0   |    0     |    -     |                                            |
+-----------------+----------+----------+--------------------------------------------+
|             1   |    1     |    0     |  GPU optical simulation only               |
+-----------------+----------+----------+--------------------------------------------+
|             2   |    1     |    1     |  CPU optical simulation only               |
+-----------------+----------+----------+--------------------------------------------+
|             3   |    2     |   0,1    |  both GPU and CPU optical simulations      |
+-----------------+----------+----------+--------------------------------------------+

**/


void SEvt::CreateOrReuse()
{
    int integrationMode = SEventConfig::IntegrationMode() ;
    LOG(LEVEL) << " integrationMode " << integrationMode  ;

    if( integrationMode == 0 )
    {
        CreateOrReuse(ECPU);          // HMM: is this needed, for metadata recording ?
    }
    else if( integrationMode == 1 )   // GPU optical simulation only
    {
        CreateOrReuse(EGPU);
    }
    else if( integrationMode == 2 )   // CPU optical simulation only
    {
        CreateOrReuse(ECPU);
    }
    else if( integrationMode == 3 )  // both CPU and GPU optical simulation
    {
        CreateOrReuse(EGPU);
        CreateOrReuse(ECPU);
    }
    else
    {
        LOG(LEVEL) << " NOT CREATING SEvt : unexpected integrationMode " << integrationMode ;
        //std::raise(SIGINT);
        //assert(0);
    }
    LOG(LEVEL) << DescINSTANCE()  ;
}



void SEvt::SetFrame(const sframe& fr )
{
    if(Exists(0)) Get(0)->setFrame(fr);
    if(Exists(1)) Get(1)->setFrame(fr);
}


void SEvt::Check(int idx)
{
    SEvt* sev = Get(idx) ;
    LOG_IF(fatal, !sev) << "must instanciate SEvt before using most SEvt methods" ;
    assert(sev);
}


#ifndef PRODUCTION
/**
SEvt::AddTag
----------------

Tags are used when recording all randoms consumed by simulation

NB some statics make sense to broadcast to all INSTANCES but
this is not one of them

**/

void SEvt::AddTag(int idx, unsigned stack, float u )
{
    if(Exists(idx)) Get(idx)->addTag(stack,u);
}
int  SEvt::GetTagSlot(int idx)
{
    return Exists(idx) ? Get(idx)->getTagSlot() : -1 ;
}
#endif

sgs SEvt::AddGenstep(const quad6& q)
{
    sgs label = {} ;
    if(Exists(0)) label = Get(0)->addGenstep(q) ;
    if(Exists(1)) label = Get(1)->addGenstep(q) ;
    return label ;
}
sgs SEvt::AddGenstep(const NP* a)
{
    sgs label = {} ;
    if(Exists(0)) label = Get(0)->addGenstep(a) ;
    if(Exists(1)) label = Get(1)->addGenstep(a) ;
    return label ;
}
void SEvt::AddCarrierGenstep(){ AddGenstep(SEvent::MakeCarrierGenstep()); }
void SEvt::AddTorchGenstep(){   AddGenstep(SEvent::MakeTorchGenstep());   }

void SEvt::addTorchGenstep()
{
    const NP* a = SEvent::MakeTorchGenstep() ;
    addGenstep(a);
}



SEvt* SEvt::LoadAbsolute(const char* dir_) // static
{
    const char* dir = spath::Resolve(dir_);
    SEvt* ev = new SEvt ;
    int rc = ev->loadfold(dir) ;
    if(rc != 0) ev->is_loadfail = true ;
    return ev  ;
}


/**
SEvt::LoadRelative (formerly Load)
------------------------------------

Q: which instance slot ?
A: are persisting instance in domain metadata and recovering in SEvt::onload

Q: should the INSTANCE array slot be filled onload ?
A: Guess so, Loading should regain the state before the save


**/

SEvt* SEvt::LoadRelative(const char* rel, int ins, int idx  )  // static
{
    LOG(LEVEL) << "[" ;

    if(rel != nullptr) SEventConfig::SetEventReldir(rel);

    SEvt* ev = SEvt::Create(ins) ;
    if(idx > -1) ev->setIndex(idx);

    int rc = ev->load() ;
    if(rc != 0) ev->is_loadfail = true ;

    LOG(LEVEL) << "]" ;
    return ev ;
}


/**
SEvt::ClearOutput
-------------------

**/

void SEvt::ClearOutput()
{
    if(Exists(0)) Get(0)->clear_output();
    if(Exists(1)) Get(1)->clear_output();
}
void SEvt::ClearGenstep()
{
    if(Exists(0)) Get(0)->clear_genstep();
    if(Exists(1)) Get(1)->clear_genstep();
}




void SEvt::Save()
{
    if(Exists(0)) Get(0)->save();
    if(Exists(1)) Get(1)->save();
}


bool SEvt::HaveDistinctOutputDirs() // static
{
    if(Count() < 2) return true ;
    assert( Count() == 2 );
    SEvt* i0 = Get(0);
    SEvt* i1 = Get(1);
    return i0->index != i1->index ;
   // NO LONGER NEEDED ? NOW THAT USE 'A' 'B' prefix ?
}


void SEvt::Save(const char* dir)
{
    assert( HaveDistinctOutputDirs() );  // check will not overwrite
    if(Exists(0)) Get(0)->save(dir) ;
    if(Exists(1)) Get(1)->save(dir) ;
}

void SEvt::Save(const char* dir, const char* rel)
{
    assert( HaveDistinctOutputDirs() );  // check will not overwrite
    if(Exists(0)) Get(0)->save(dir, rel) ;
    if(Exists(1)) Get(1)->save(dir, rel) ;
}

void SEvt::SaveGenstepLabels(const char* dir, const char* name)
{
    assert( HaveDistinctOutputDirs() );  // check will not overwrite
    if(Exists(0)) Get(0)->saveGenstepLabels(dir, name );
    if(Exists(1)) Get(1)->saveGenstepLabels(dir, name );
}





void SEvt::BeginOfRun()
{
    SetRunProf("SEvt__BeginOfRun");
    SaveRunMeta();  // HMM: for crash debug, RunMeta is saved at BeginOfRun EndOfRun and endOfEvent
}




void SEvt::EndOfRun()
{
    SetRunProf("SEvt__EndOfRun");
    SaveRunMeta();

    if(EndOfRun_SProf == 1 )
    {
        bool append = false ;
        SProf::Write("SEvt__EndOfRun_SProf.txt", append ) ;
    }
}


const int SEvt::EndOfRun_SProf = ssys::getenvint("SEvt__EndOfRun_SProf",-1) ;



template<typename T>
void SEvt::SetRunMeta(const char* k, T v )
{
    RUN_META->set_meta<T>(k, v );
}

template void SEvt::SetRunMeta<int>(      const char*, int  );
template void SEvt::SetRunMeta<uint64_t>( const char*, uint64_t  );
template void SEvt::SetRunMeta<int64_t>(  const char*, int64_t  );
template void SEvt::SetRunMeta<unsigned>( const char*, unsigned  );
template void SEvt::SetRunMeta<float>(    const char*, float  );
template void SEvt::SetRunMeta<double>(   const char*, double  );
template void SEvt::SetRunMeta<std::string>( const char*, std::string  );

void SEvt::SetRunMetaString(const char* k, const char* v ) // static
{
    std::string* rms = RunMetaString();
    assert(rms);
    NP::SetMeta<std::string>(*rms, k, v );
}



void SEvt::SetRunProf(const char* k, const sprof& v) // static
{
    SetRunMeta<std::string>( k, sprof::Serialize(v) );
}
void SEvt::SetRunProf(const char* k)   // static
{
    SetRunMeta<std::string>( k, sprof::Now() );
}

void SEvt::setRunProf_Annotated(const char* hdr) const
{
    std::string eid = getIndexString_(hdr) ;
    SetRunMeta<std::string>( eid.c_str(), sprof::Now() );
}




/**
SEvt::IsSaveNothing
--------------------

When using OPTICKS_EVENT_MODE of Minimal or Nothing
the saving of run metadata and directory creation
can be disabled with::

    export SEvt__SAVE_NOTHING=1

**/

bool SEvt::IsSaveNothing() // static
{
    return SEventConfig::IsMinimalOrNothing() && SAVE_NOTHING ;
}



/**
SEvt::SaveRunMeta
-------------------

Saving only at the end of run is problematic
when jobs are prone to memory failure. So
better to save at the end of every event, not
just the last.

**/

void SEvt::SaveRunMeta(const char* base)
{
    const char* dir = RunDir(base);

    bool is_save_nothing = IsSaveNothing();

    LOG_IF(info, RUNMETA)
        << " [" << SEvt__RUNMETA << "]"
        << " is_save_nothing " << ( is_save_nothing ? "YES" : "NO " )
        << " base " << ( base ? base : "-" )
        << " dir " << ( dir ? dir : "-" )
        ;

    if(!is_save_nothing) RUN_META->save(dir, "run.npy") ;
}

void SEvt::setMetaString(const char* k, const char* v)
{
    NP::SetMeta<std::string>(meta, k, v );
}

void SEvt::setMetaProf(const char* k, const sprof& v)
{
    NP::SetMeta<std::string>(meta, k, sprof::Serialize(v) );
}
void SEvt::setMetaProf(const char* k)
{
    NP::SetMeta<std::string>(meta, k, sprof::Now() );
}





template<typename T>
void SEvt::setMeta( const char* k, T v )
{
    NP::SetMeta<T>( meta, k, v );
}

template void SEvt::setMeta<uint64_t>(const char*, uint64_t );
template void SEvt::setMeta<int>(const char*, int );
template void SEvt::setMeta<unsigned>(const char*, unsigned );
template void SEvt::setMeta<float>(const char*, float );
template void SEvt::setMeta<double>(const char*, double );
template void SEvt::setMeta<std::string>(const char*, std::string );



/**
SEvt::beginOfEvent
-------------------

Called from::

     QSim::simulate (SEvt::EGPU instance )
     U4Recorder::BeginOfEventAction  (SEvt::ECPU instance)

Note that eventID from both Geant4 and now Opticks is zero based

Lifecycle::

      G4CXApp::BeginOfEventAction
        U4Recorder::BeginOfEventAction
          ECPU.SEvt::beginOfEvent

      ... collect gensteps into whichever SEvt are instanciated
      ... collect gensteps into whichever SEvt are instanciated
      ... collect gensteps into whichever SEvt are instanciated

      G4CXApp::EndOfEventAction
        U4Recorder::EndOfEventAction
          ECPU.SEvt::endOfEvent

        G4CXOpticks::simulate
          QSim::simulate

            EGPU.SEvt::beginOfEvent

            QEvent::setGenstep
               A:SEvt::clear
               QEvent::setGenstep(NP*)

            SCSGOptiX::simulate_launch

            EGPU.:SEvt::endOfEvent


ECPU
-----

As gensteps are collected before EGPU.beginOfEvent
the clear_output excludes gs/gensteps as they are inputs.

Need to think of the lifecycle of both ECPU and EGPU together.
This remains true even with runningMode 1 which has no ECPU
as still need to collect the gensteps.

**/


void SEvt::beginOfEvent(int eventID)
{
    if(isFirstEvtInstance() && eventID == 0) BeginOfRun() ;
    if(eventID == 0) SetRunProf( isEGPU() ? "SEvt__beginOfEvent_FIRST_EGPU" : "SEvt__beginOfEvent_FIRST_ECPU" ) ;

    setStage(SEvt__beginOfEvent);
    sprof::Stamp(p_SEvt__beginOfEvent_0);

    LOG(LEVEL) << " eventID " << eventID ;   // 0-based
    setIndex(eventID);

    LOG_IF(info, LIFECYCLE) << id() ;

    clear_output();   // output vectors and fold : excluding gensteps as thats input
    if( addGenstep_array == 0 )
    {
        addInputGenstep();  // does genstep setup for simtrace, input photon and torch running
    }
    else
    {
        LOG(info) << "skip addInputGenstep as addGenstep_array " << addGenstep_array ;
    }


    setMeta<int>("NumPhotonCollected", numphoton_collected );
    setMeta<int>("NumGenstepCollected", numgenstep_collected );
    setMeta<int>("MaxBounce", evt->max_bounce );

    LOG_IF(info, LIFECYCLE)
        << " NumPhotonCollected " << numphoton_collected
        << " NumGenstepCollected " << numgenstep_collected
        << " MaxBounce " << evt->max_bounce
        ;

    sprof::Stamp(p_SEvt__beginOfEvent_1);
}



/**
SEvt::endOfEvent
------------------

Called for example from::

   U4Recorder::EndOfEventAction
   QSim::simulate

Only SEventConfig::SaveComp OPTICKS_SAVE_COMP are actually saved,
so can switch off all saving wuth that config.

**/

void SEvt::endOfEvent(int eventID)
{

    setStage(SEvt__endOfEvent);
    LOG_IF(info, LIFECYCLE) << id() ;
    sprof::Stamp(p_SEvt__endOfEvent_0);

    endIndex(eventID);   // eventID is 0-based
    endMeta();
    gather_metadata();

    save();              // gather and save SEventConfig configured arrays
    clear_output();
    clear_genstep();
    clear_extra();
    reset_counter();

    SaveRunMeta(); // saving run_meta.txt at end of every event incase of crashes


    bool is_last_eventID = SEventConfig::IsLastEvent(eventID) ;
    if(is_last_eventID)
    {
        //SetRunProf( isEGPU() ? "SEvt__endOfEvent_LAST_EGPU" : "SEvt__endOfEvent_LAST_ECPU" ) ;
        bool is_last_evt_instance = isLastEvtInstance() ;

        LOG(LEVEL)
            << " is_last_eventID " << ( is_last_eventID ? "YES" : "NO " )
            << " is_last_evt_instance " << ( is_last_evt_instance ? "YES" : "NO " )
            ;

        if(is_last_evt_instance) SEvt::EndOfRun();   // invokes SaveRunMeta
    }


}

void SEvt::reset_counter()
{
    addGenstep_array = 0  ;
}


void SEvt::endMeta()
{
    setMeta<std::string>("site", "SEvt::endMeta" );
    setMeta<int>("hitmask", selector->hitmask );
    setMeta<int>("index", index);
    setMeta<int>("instance", instance);

    setMetaProf("SEvt__beginOfEvent_0", p_SEvt__beginOfEvent_0);
    setMetaProf("SEvt__beginOfEvent_1", p_SEvt__beginOfEvent_1);
    setMetaProf("SEvt__endOfEvent_0",   p_SEvt__endOfEvent_0);
    //setMetaProf("SEvt__endOfEvent_1",   p_SEvt__endOfEvent_1);

    setMeta<uint64_t>("t_BeginOfEvent", t_BeginOfEvent );

#ifndef PRODUCTION
    setMeta<uint64_t>("t_setGenstep_0",  t_setGenstep_0 );
    setMeta<uint64_t>("t_setGenstep_1",  t_setGenstep_1 );
    setMeta<uint64_t>("t_setGenstep_2",  t_setGenstep_2 );
    setMeta<uint64_t>("t_setGenstep_3",  t_setGenstep_3 );
    setMeta<uint64_t>("t_setGenstep_4",  t_setGenstep_4 );
    setMeta<uint64_t>("t_setGenstep_5",  t_setGenstep_5 );
    setMeta<uint64_t>("t_setGenstep_6",  t_setGenstep_6 );
    setMeta<uint64_t>("t_setGenstep_7",  t_setGenstep_7 );
    setMeta<uint64_t>("t_setGenstep_8",  t_setGenstep_8 );
#endif
    setMeta<uint64_t>("t_PreLaunch", t_PreLaunch );
    setMeta<uint64_t>("t_PostLaunch", t_PostLaunch );
    setMeta<uint64_t>("t_EndOfEvent",   t_EndOfEvent );

    setMeta<uint64_t>("t_Event", t_EndOfEvent - t_BeginOfEvent );
    setMeta<double>("t_Launch", t_Launch );
}



int SEvt::GetIndex(int idx)
{
    return Exists(idx) ? Get(idx)->getIndex() : -1 ;
}
S4RandomArray* SEvt::GetRandomArray(int idx)
{
    return Exists(idx) ? Get(idx)->random_array : nullptr ;
}


int SEvt::GetNumPhotonCollected(int idx){    return Exists(idx) ? Get(idx)->getNumPhotonCollected() : UNDEF ; }
int SEvt::GetNumPhotonGenstepMax(int idx){   return Exists(idx) ? Get(idx)->getNumPhotonGenstepMax() : UNDEF ; }
int SEvt::GetNumPhotonFromGenstep(int idx){  return Exists(idx) ? Get(idx)->getNumPhotonFromGenstep() : UNDEF ; }
int SEvt::GetNumGenstepFromGenstep(int idx){ return Exists(idx) ? Get(idx)->getNumGenstepFromGenstep() : UNDEF ; }
int SEvt::GetNumHit(int idx){                return Exists(idx) ? Get(idx)->getNumHit() : UNDEF ; }
int SEvt::GetNumHit_EGPU(){  return GetNumHit(EGPU) ; }
int SEvt::GetNumHit_ECPU(){  return GetNumHit(ECPU) ; }



NP* SEvt::GatherGenstep(int idx) {   return Exists(idx) ? Get(idx)->gatherGenstep() : nullptr ; }
NP* SEvt::GetInputPhoton(int idx) {  return Exists(idx) ? Get(idx)->getInputPhoton() : nullptr ; }

void SEvt::SetInputPhoton(NP* p)
{
    if(Exists(0)) Get(0)->setInputPhoton(p) ;
    if(Exists(1)) Get(1)->setInputPhoton(p) ;
}
bool SEvt::HasInputPhoton(int idx)
{
    return Exists(idx) ? Get(idx)->hasInputPhoton() : false ;
}
bool SEvt::HasInputPhoton()
{
    return HasInputPhoton(EGPU) || HasInputPhoton(ECPU) ;
}

NP* SEvt::GetInputPhoton() // static
{
    NP* ip = nullptr ;
    if(ip == nullptr && HasInputPhoton(EGPU)) ip = GetInputPhoton(EGPU) ;
    if(ip == nullptr && HasInputPhoton(ECPU)) ip = GetInputPhoton(ECPU) ;
    return ip ;
}


std::string SEvt::DescHasInputPhoton()  // static
{
    std::stringstream ss ;
    ss
       <<  "SEvt::DescHasInputPhoton()  "
       << " SEventConfig::IntegrationMode " << SEventConfig::IntegrationMode()
       << " SEvt::HasInputPhoton(EGPU) " << HasInputPhoton(EGPU)
       << " SEvt::HasInputPhoton(ECPU) " << HasInputPhoton(ECPU)
       << std::endl
       << "SEvt::Brief"
       << std::endl
       << SEvt::Brief()
       << std::endl
       << "SEvt::DescInputPhoton(EGPU)"
       << SEvt::DescInputPhoton(EGPU)
       << "SEvt::DescInputPhoton(ECPU)"
       << SEvt::DescInputPhoton(ECPU)
       << std::endl
       ;
    std::string str = ss.str();
    return str ;
}




/**
SEvt::clear_genstep_vector
----------------------------

1. set photon counts to zero
2. clears gs and genstep vectors



**/

void SEvt::clear_genstep_vector()
{
    numgenstep_collected = 0u ;
    numphoton_collected = 0u ;
    numphoton_genstep_max = 0u ;

    clear_genstep_vector_count += 1 ;

    setNumPhoton(0);

    gs.clear();
    genstep.clear();
    gather_done = false ;
}



/**
SEvt::clear_output_vector
---------------------------


Notice

1. no hit : thats a sub-selection of the photon
2. genstep+gs vectors are not cleared : they are inputs to the optical simulation, not outputs

Note that most of the vectors are only used with hostside running.

**/


void SEvt::clear_output_vector()
{
    clear_output_vector_count += 1 ;

    pho.clear();
    slot.clear();
    photon.clear();
    record.clear();
    rec.clear();
    seq.clear();
    prd.clear();
    tag.clear();
    flat.clear();
    simtrace.clear();
    aux.clear();
    sup.clear();
    g4state = nullptr ;   // avoiding stale (g4state is special, as only used for 1st event)
}





/**
SEvt::clear_output
--------------------

* Called from SEvt::beginOfEvent and SEvt::endOfEvent
* Clear output vectors and the fold excluding the gensteps.


**/

void SEvt::clear_output()
{
    setStage(SEvt__clear_output);

    LOG_IF(info, LIFECYCLE) << id() << " BEFORE clear_output_vector " ;

    clear_output_vector();

    const char* keylist = "genstep" ;
    bool copy = false ;
    char delim = ',' ;

    topfold->clear_except(keylist, copy, delim );

    LOG_IF(info, LIFECYCLE) << id() << " AFTER clear_output_vector " ;

    LOG(LEVEL) << "]" ;
}

void SEvt::clear_extra()
{
    LOG_IF(info, LIFECYCLE) << id() << " BEFORE extrafold.clear " ;

    extrafold->clear();

    LOG_IF(info, LIFECYCLE) << id() << " AFTER extrafold.clear " ;
}


/**
SEvt::clear_genstep
---------------------

* canonical call from SEvt::endOfEvent

**/


void SEvt::clear_genstep()
{
    setStage(SEvt__clear_genstep);
    LOG_IF(info, LIFECYCLE) << id() << " BEFORE clear_genstep_vector " ;

    clear_genstep_vector();
    topfold->clear_only("genstep", false, ',');

    LOG_IF(info, LIFECYCLE) << id() << " AFTER clear_genstep_vector " ;
}


/**
SEvt::setIndex
---------------

index_arg
    zero based event index expected to come from Geant4 eventID for example

index
    event index obtained from SEventConfig::EventIndex(index_arg)
    which may be offset by the OPTICKS_START_INDEX envvar

**/

void SEvt::setIndex(int index_arg)
{
    assert( index_arg >= 0 );
    index = SEventConfig::EventIndex(index_arg) ;
    t_BeginOfEvent = sstamp::Now();                // moved here from the static

    setRunProf_Annotated("SEvt__setIndex_" );
}
void SEvt::endIndex(int index_arg)
{
    int index_expected = SEventConfig::EventIndex(index_arg) ;
    bool consistent = index_expected == index ;
    LOG_IF(fatal, !consistent)
         << " index_arg " << index_arg
         << " index_expected " << index_expected
         << " index " << index
         << " consistent " << ( consistent ? "YES" : "NO " )
         ;
    assert( consistent );
    t_EndOfEvent = sstamp::Now();

    setRunProf_Annotated("SEvt__endIndex_" );
}

/**
SEvt::getIndexArg
------------------

0-based index in range 0..num-1 even when a non-zero startIndex offset is in use

**/

int SEvt::getIndexArg() const
{
    return SEventConfig::EventIndexArg(index);
}
int SEvt::getIndex() const
{
    return index ;
}
int SEvt::getIndexPresentation() const
{
    return index == MISSING_INDEX ? -1 : index ;
}

std::string SEvt::descIndex() const
{
    std::stringstream ss ;
    ss << "SEvt::descIndex"
       << " index                " << index                  << " (internal index which may be offset by OPTICKS_START_INDEX) " << std::endl
       << " getIndexPresentation " << getIndexPresentation() << " (internal index presentation avoidsing MISSING_INDEX " << std::endl
       << " getIndexArg          " << getIndexArg()          << " (external index removing any offsets, mapping back to eg Geant4 eventID " << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}

void SEvt::incrementIndex()
{
    int index_arg = getIndexArg();  // -1 when MISSING_INDEX
    setIndex(index_arg + 1);
}
void SEvt::unsetIndex()
{
    index = MISSING_INDEX ;
}



void SEvt::setInstance(int instance_)
{
    instance = instance_ ;
}
int SEvt::getInstance() const
{
    return instance ;
}




/**
SEvt::getNumGenstepFromGenstep
--------------------------------

Size of the genstep vector ...

caution this vector is cleared by QEvent::setGenstep
after which must get the count from the genstep array

**/

unsigned SEvt::getNumGenstepFromGenstep() const
{
    assert( genstep.size() == gs.size() );
    return genstep.size() ;
}

/**
SEvt::getNumPhotonFromGenstep
----------------------------------

Total photons by summation over all collected genstep.
When collecting very large numbers of gensteps the alternative
SEvt::getNumPhotonCollected is faster.

**/

unsigned SEvt::getNumPhotonFromGenstep() const
{
    unsigned tot = 0 ;
    for(unsigned i=0 ; i < genstep.size() ; i++) tot += genstep[i].numphoton() ;
    return tot ;
}

unsigned SEvt::getNumGenstepCollected() const
{
    return numgenstep_collected ;  // updated by addGenstep
}
unsigned SEvt::getNumPhotonCollected() const
{
    return numphoton_collected ;   // updated by addGenstep
}
unsigned SEvt::getNumPhotonGenstepMax() const
{
    return numphoton_genstep_max ;
}



/**
SEvt::addGenstep
------------------

The sgs summary struct of the last genstep added is returned.

**/

sgs SEvt::addGenstep(const NP* a)
{
    sgs s = {} ;

    if( a == nullptr )
    {
        LOG(error) << " a null : low level simtrace tests like CSGSimtraceTest.sh can do this  " ;
        return s ;
    }

    assert( addGenstep_array == 0 );
    addGenstep_array++ ;

    int num_gs = a ? a->shape[0] : -1 ;
    assert( num_gs > 0 );
    quad6* qq = (quad6*)a->bytes();
    for(int i=0 ; i < num_gs ; i++) s = addGenstep(qq[i]) ;

    if(SEventConfig::IsRGModeSimtrace() && SFrameGenstep::HasConfigEnv()) // CEGS running
    {
        if(frame.is_hostside_simtrace()) setFrame_HostsideSimtrace();
    }

    return s ;
}


/**
SEvt::addGenstep
------------------

The GIDX envvar is used whilst debugging to restrict to collecting
a single genstep chosen by its index.  This is implemented by
always collecting all genstep labels, but only collecting
actual gensteps for the enabled index.

In addition to appending the quad6 to the genstep vector this also
does some bookkeeping, updating::

1. numphoton_collected
2. numgenstep_collected

Also SEvt::setNumPhoton called, configuring the sizes which
get allocated later.

**/


sgs SEvt::addGenstep(const quad6& q_)
{
    LOG_IF(info, LIFECYCLE) << id() ;
    dbg->addGenstep++ ;
    LOG(LEVEL) << " index " << index << " instance " << instance ;

    unsigned gentype = q_.gentype();
    unsigned matline_ = q_.matline();

    //bool is_input_photon_gs = OpticksGenstep_::IsInputPhoton(gentype) ;
    bool is_cerenkov_gs = OpticksGenstep_::IsCerenkov(gentype);


    /*
    bool input_photon_with_normal_genstep = input_photon && !is_input_photon_gs  ;
    LOG_IF(fatal, input_photon_with_normal_genstep)
        << "input_photon_with_normal_genstep " << input_photon_with_normal_genstep
        << " MIXING input photons with other gensteps is not allowed "
        << " for example avoid defining OPTICKS_INPUT_PHOTON when doing simtrace"
        ;
    assert( input_photon_with_normal_genstep == false );
    */



    int gidx = int(gs.size())  ;  // 0-based genstep label index
    bool enabled = GIDX == -1 || GIDX == gidx ;

    quad6& q = const_cast<quad6&>(q_);
    if(!enabled) q.set_numphoton(0);
    // simplify handling of disabled gensteps by simply setting numphoton to zero for them


    if(matline_ >= G4_INDEX_OFFSET  )
    {
        unsigned mtindex = matline_ - G4_INDEX_OFFSET ;
        int matline = cf ? cf->lookup_mtline(mtindex) : 0 ;    // cf(SGeo) used for lookup

        bool bad_ck = is_cerenkov_gs && matline == -1 ;

        LOG_IF(info, bad_ck )
            << " is_cerenkov_gs " << ( is_cerenkov_gs ? "YES" : "NO " )
            << " cf " << ( cf ? "YES" : "NO " )
            << " bad_ck "
            << " matline_ " << matline_
            << " matline " << matline
            << " gentype " << gentype
            << " mtindex " << mtindex
            << " G4_INDEX_OFFSET " << G4_INDEX_OFFSET
            << " desc_mt "
            << std::endl
            << ( cf ? cf->desc_mt() : "no-cf" )
            << std::endl
            ;

        q.set_matline(matline);  // <=== THIS IS CHANGING GS BACK IN CALLERS SCOPE

    }


#ifdef SEVT_NUMPHOTON_FROM_GENSTEP_CHECK
    unsigned numphoton_from_genstep = getNumPhotonFromGenstep() ; // sum numphotons from all previously collected gensteps (since last clear)
    assert( numphoton_from_genstep == numphoton_collected );
#endif

    unsigned q_numphoton = q.numphoton() ;          // numphoton in this genstep
    if(q_numphoton > numphoton_genstep_max) numphoton_genstep_max = q_numphoton ;


    sgs s = {} ;                      // genstep summary struct

    s.index = genstep.size() ;        // 0-based genstep index since last clear
    s.photons = q_numphoton ;         // numphoton in this genstep
    s.offset = numphoton_collected ;  // sum numphotons from all previously collected gensteps (since last clear)
    s.gentype = q.gentype() ;

    gs.push_back(s) ;                 // summary labels
    genstep.push_back(q) ;            // actual genstep params : copied into the vector

    numgenstep_collected += 1 ;
    numphoton_collected += q_numphoton ;  // keep running total for all gensteps collected, since last clear


    int tot_photon = s.offset+s.photons ;

    LOG_IF(debug, enabled) << " s.desc " << s.desc() << " gidx " << gidx << " enabled " << enabled << " tot_photon " << tot_photon ;

    bool num_photon_changed = tot_photon != evt->num_photon ;

    LOG(LEVEL)
        << " tot_photon " << tot_photon
        << " evt.num_photon " << evt->num_photon
        << " num_photon_changed " << num_photon_changed
        << " gs.size " << gs.size()
        << " genstep.size " << genstep.size()
        << " numgenstep_collected " << numgenstep_collected
        << " numphoton_collected " << numphoton_collected
        << " tot_photon " << tot_photon
        << " s.index " << s.index
        << " s.photons " << s.photons
        << " s.offset " << s.offset
        << " s.gentype " << s.gentype
        << " s.desc " << s.desc()
        ;

    setNumPhoton(tot_photon); // still call when no change for reset hostside_running_resize_done:false
    return s ;
}



/**
SEvt::setNumPhoton
----------------------

NB no allocations are done here, in part because this is called on adding every genstep.
This is just preparing array sizes for future array allocations.


This is called from SEvt::addGenstep, updating evt.num_photon
according to the additional genstep collected and evt.num_seq/tag/flat/record/rec/prd
depending on the configured max which when zero will keep the counts zero.

Also called by QEvent::setNumPhoton prior to device side allocations.

*hostside_running_resize_done:false*
    signals the following SEvt::beginPhoton to call SEvt::hostside_running_resize,
    so allocation for hostside running happens on reaching the first photon track

Note that SEvt::beginPhoton is used for hostside running only (eg U4Recorder/U4SimulateTest)
so as gensteps are added with SEvt::addGenstep from the U4 scintillation
and Cerenkov collection SEvt::setNumPhoton increments the totals in sevent.h:evt
and sets hostside_running_resize_done:false such that at the next SEvt::beginPhoton
which happens from the BeginOfTrackAction the std::vector grow as
needed to accommodate the photons from the last genstep collected.

**/

void SEvt::setNumPhoton(unsigned num_photon)
{
    //LOG_IF(info, LIFECYCLE) << id() << " num_photon " << num_photon ;
    bool num_photon_allowed = num_photon <= (unsigned)(evt->max_photon) ;   // evt->max_slot ?
    const int M = 1000000 ;

    LOG_IF(fatal, !num_photon_allowed)
        << " num_photon/M " << num_photon/M
        << " evt.max_photon/M " << evt->max_photon/M
        << " num_photon " << num_photon
        << " evt.max_photon " << evt->max_photon
        ;
    assert( num_photon_allowed );

    LOG_IF(info, INDEX)
        << " set SEvt::index to sevent::index " << index
        << " num_photon " << num_photon
        ;

    evt->index = index ;

    evt->num_photon = num_photon ;
    evt->num_seq    = evt->max_seq  == 1 ? evt->num_photon : 0 ;
    evt->num_tag    = evt->max_tag  == 1 ? evt->num_photon : 0 ;
    evt->num_flat   = evt->max_flat == 1 ? evt->num_photon : 0 ;
    evt->num_sup    = evt->max_sup   > 0 ? evt->num_photon : 0 ;

    evt->num_record = evt->max_record * evt->num_photon ;
    evt->num_rec    = evt->max_rec    * evt->num_photon ;
    evt->num_aux    = evt->max_aux    * evt->num_photon ;
    evt->num_prd    = evt->max_prd    * evt->num_photon ;

    LOG(LEVEL)
        << " evt->num_photon " << evt->num_photon
        << " evt->num_tag " << evt->num_tag
        << " evt->num_flat " << evt->num_flat
        ;

    hostside_running_resize_done = false ;
}




/**
SEvt::setNumSimtrace
---------------------

**/

void SEvt::setNumSimtrace(unsigned num_simtrace)
{
    bool num_simtrace_allowed = int(num_simtrace) <= evt->max_simtrace ;
    LOG_IF(fatal, !num_simtrace_allowed) << " num_simtrace " << num_simtrace << " evt.max_simtrace " << evt->max_simtrace ;
    assert( num_simtrace_allowed );
    LOG(LEVEL) << " num_simtrace " << num_simtrace ;

    evt->num_simtrace = num_simtrace ;
}


/**
SEvt::hostside_running_resize
-------------------------------

Canonically called from SEvt::beginPhoton  (also SEvt::setFrame_HostsideSimtrace)

**/

void SEvt::hostside_running_resize()
{
    bool is_self_provider = isSelfProvider() ;
    LOG_IF(fatal, is_self_provider == false ) << " NOT-is_self_provider " << descProvider() ;
    LOG(LEVEL)
        << " is_self_provider " << is_self_provider
        << " hostside_running_resize_done " << hostside_running_resize_done
        ;

    assert( hostside_running_resize_done == false );
    assert( is_self_provider );

    hostside_running_resize_done = true ;
    hostside_running_resize_();

    LOG(LEVEL)
        << " is_self_provider " << is_self_provider
        << std::endl
        << evt->desc()
        ;

}

/**
SEvt::hostside_running_resize_
--------------------------------

Resize the hostside std::vectors according to the sizes from sevent.h:evt
and update evt array pointers to follow around the std::vectors as they get reallocated.

This makes the hostside sevent.h:evt environment mimic the deviceside environment
even though deviceside uses device buffers and hostside uses std::vectors.

Notice how the same sevent.h struct that holds deviceside pointers
is being using to hold the hostside vector data pointers.


+-----------+-----------------+
| vector    |   evt->         |
+===========+=================+
| pho       |   num_photon    |
+-----------+-----------------+
| slot      |   num_photon    |
+-----------+-----------------+
| photon    |   num_photon    |
+-----------+-----------------+
| record    |   num_record    |
+-----------+-----------------+
| rec       |   num_rec       |
+-----------+-----------------+
| aux       |   num_aux       |
+-----------+-----------------+
| seq       |   num_seq       |
+-----------+-----------------+
| prd       |   num_prd       |
+-----------+-----------------+
| tag       |   num_tag       |
+-----------+-----------------+
| flat      |   num_flat      |
+-----------+-----------------+
| simtrace  |   num_simtrace  |
+-----------+-----------------+

Vectors are disabled by some of the above *num*
being configured to zero via SEventConfig.

**/

void SEvt::hostside_running_resize_()
{
    LOG(LEVEL)
        << " photon.size " << photon.size()
        << " photon.size/M " << photon.size()/M
        << " => "
        << " evt.num_photon " << evt->num_photon
        << " evt.num_photon/M " << evt->num_photon/M
        ;

    bool shrink = true ;

    if(evt->num_photon > 0)        // no device side equivalent array
    {
        pho.resize(  evt->num_photon );
        if(shrink) pho.shrink_to_fit();
    }
    if(evt->num_photon > 0)       // no device side equivalent array
    {
        slot.resize( evt->num_photon );
        if(shrink) slot.shrink_to_fit();
    }
    // pho and slot look essential for U4Recorder bookkeeping


    if(evt->num_photon > 0)
    {
        photon.resize(evt->num_photon);
        if(shrink) photon.shrink_to_fit();
        evt->photon = photon.data() ;
    }
    if(evt->num_record > 0)
    {
        record.resize(evt->num_record);
        if(shrink) record.shrink_to_fit();
        evt->record = record.data() ;
    }
    if(evt->num_rec > 0)
    {
        rec.resize(evt->num_rec);
        if(shrink) rec.shrink_to_fit();
        evt->rec = rec.data() ;
    }
    if(evt->num_aux > 0)
    {
        aux.resize(evt->num_aux);
        if(shrink) aux.shrink_to_fit();
        evt->aux = aux.data() ;
    }
    if(evt->num_sup > 0)
    {
        sup.resize(evt->num_sup);
        if(shrink) sup.shrink_to_fit();
        evt->sup = sup.data() ;
    }
    if(evt->num_seq > 0)
    {
        seq.resize(evt->num_seq);
        if(shrink) seq.shrink_to_fit();
        evt->seq = seq.data() ;
    }
    if(evt->num_prd > 0)
    {
        prd.resize(evt->num_prd);
        if(shrink) prd.shrink_to_fit();
        evt->prd = prd.data() ;
    }
    if(evt->num_tag > 0)
    {
        tag.resize(evt->num_tag);
        if(shrink) tag.shrink_to_fit();
        evt->tag = tag.data() ;
    }
    if(evt->num_flat > 0)
    {
        flat.resize(evt->num_flat);
        if(shrink) flat.shrink_to_fit();
        evt->flat = flat.data() ;
    }
    if(evt->num_simtrace > 0)
    {
        simtrace.resize(evt->num_simtrace);
        if(shrink) simtrace.shrink_to_fit();
        evt->simtrace = simtrace.data() ;
    }
}

/**
SEvt::get_gs
--------------

Lookup sgs genstep label corresponding to spho photon label

**/

const sgs& SEvt::get_gs(const spho& label) const
{
    assert( label.gs < int(gs.size()) );
    const sgs& _gs =  gs[label.gs] ;
    return _gs ;
}

/**
SEvt::get_genflag
-------------------

This is called for example from SEvt::beginPhoton
to become the "TO" "SI" "CK" at the start of photon histories.

1. lookup sgs genstep label corresponding to the spho photon label
2. convert the sgs gentype into the corresponding photon genflag which must be
   one of CERENKOV, SCINTILLATION or TORCH

   * What about input photons ? Presumably they are TORCH ?

**/

unsigned SEvt::get_genflag(const spho& label) const
{
    const sgs& _gs = get_gs(label);
    int gentype = _gs.gentype ;
    unsigned genflag = OpticksGenstep_::GenstepToPhotonFlag(gentype);
    assert( genflag == CERENKOV || genflag == SCINTILLATION || genflag == TORCH );
    return genflag ;
}


/**
SEvt::beginPhoton : only used for hostside running eg with G4CXTest
---------------------------------------------------------------------------

Canonically invoked from tail of U4Recorder::PreUserTrackingAction_Optical

0. for hostside_running_resize_done:false calls hostside_running_resize which resizes vectors
   and updates evt pointers : via the toggle this only gets done when reaching first photon of the event
1. determine genflag SI/CK/TO, via lookups in the sgs corresponding to the spho label
2. zeros current_ctx and slot[label.id] (the "recording head" )
3. start filling current_ctx.p sphoton with set_idx and set_flag

**/
void SEvt::beginPhoton(const spho& label)
{
    if(!hostside_running_resize_done) hostside_running_resize();

    dbg->beginPhoton++ ;
    LOG(LEVEL) ;
    LOG(LEVEL) << label.desc() ;

    unsigned idx = label.id ;

    bool in_range = idx < pho.size() ;
    LOG_IF(error, !in_range)
        << " not in_range "
        << " idx " << idx
        << " pho.size  " << pho.size()
        << " label " << label.desc()
        ;

    if(!in_range) std::cerr
        << "SEvt::beginPhoton FATAL not in_range"
        << " idx " << idx
        << " pho.size  " << pho.size()
        << " label " << label.desc()
        << std::endl
        ;

    assert(in_range);

    unsigned genflag = get_genflag(label);

    pho[idx] = label ;        // slot in the photon label
    slot[idx] = 0 ;           // slot/bounce incremented only at tail of SEvt::pointPhoton

    current_pho = label ;
    current_prd.zero() ;

    sctx& ctx = current_ctx ;
    ctx.zero();

#ifndef PRODUCTION
#ifdef WITH_SUP
    quadx6& xsup = (quadx6&)ctx.sup ;
    xsup.q0.w.x = sstamp::Now();
#endif
#endif

    ctx.idx = idx ;
    ctx.evt = evt ;
    ctx.prd = &current_prd ;   // current_prd is populated by InstrumentedG4OpBoundaryProcess::PostStepDoIt

    ctx.p.set_idx(idx);
    ctx.p.set_flag(genflag);

    bool flagmask_one_bit = ctx.p.flagmask_count() == 1  ;
    LOG_IF(error, !flagmask_one_bit )
        << " not flagmask_one_bit "
        << " : should only be a single bit in the flagmask at this juncture "
        << " label " << label.desc()
         ;

    assert( flagmask_one_bit );
}

unsigned SEvt::getCurrentPhotonIdx() const { return current_pho.id ; }


/**
SEvt::resumePhoton
---------------------

Canonically called from tail of U4Recorder::PreUserTrackingAction_Optical
following a FastSim->SlowSim transition.

Transitions between Geant4 "SlowSim" and FastSim result in fSuspend track status
and the history of a single photon being split across multiple calls to
U4RecorderTest::PreUserTrackingAction (U4Recorder::PreUserTrackingAction_Optical)

Need to join up the histories, a bit like rjoin does for reemission.
But with FastSim/SlowSim the G4Track are the same pointers, unlike
with reemission where the G4Track pointers differ.

BUT reemission photons will also go thru FastSim/SlowSim transitions,
so need separate approach than the label.gn (photon generation) handling of reemission ?

**/

void SEvt::resumePhoton(const spho& label)
{
    dbg->resumePhoton++ ;
    LOG(LEVEL);
    LOG(LEVEL) << label.desc() ;

    unsigned idx = label.id ;
    bool idx_expect = idx < pho.size() ;
    if(!idx_expect ) std::raise(SIGINT) ;
    assert( idx_expect );
}


/**
SEvt::rjoin_resumePhoton
-------------------------

Invoked from U4Recorder::PreUserTrackingAction_Optical for
reemission photons that subsequently transition from FastSim
back to SlowSim.

BUT as the FastSim/SlowSim transition happens inside PMT
will never need to do real rjoin history rewriting  ?

**/

void SEvt::rjoin_resumePhoton(const spho& label)
{
    dbg->rjoin_resumePhoton++ ;
    LOG(LEVEL);
    LOG(LEVEL) << label.desc() ;

    unsigned idx = label.id ;
    bool idx_expect = idx < pho.size() ;
    if(!idx_expect ) std::raise(SIGINT) ;
    assert( idx_expect );
}


/**
SEvt::rjoinPhoton : only used for hostside running
---------------------------------------------------

Called from tail of U4Recorder::PreUserTrackingAction_Optical for G4Track with
spho label indicating a reemission generation greater than zero.

HUH: BUT THAT MEANS THIS WILL BE CALLED FOR EVERY SUBSEQUENT
STEP OF THE REEMISSION PHOTON NOT JUST THE FIRST THAT NEEDS REJOINING ?

* SUSPECT THAT WILL GET REPEATED "RE RE RE RE" ?
* MAYBE NEED ANOTHER MARK TO SAY THE REJOIN WAS DONE AND DOENST NEED RE-rjoin ?

Note that this will mostly be called for photons that originate from
scintillation gensteps BUT it will also happen for Cerenkov (and Torch) genstep
generated photons within a scintillator due to reemission.

HMM: could directly change photon[idx] via ref ?
But are here taking a copy to current_photon
and relying on copyback at SEvt::finalPhoton

**/
void SEvt::rjoinPhoton(const spho& label)
{
    dbg->rjoinPhoton++ ;
    LOG(LEVEL);
    LOG(LEVEL) << label.desc() ;

    unsigned idx = label.id ;
    bool in_range = idx < pho.size() ;
    LOG_IF(fatal, !in_range )
         << " NOT-in_range "
         << " idx " << idx
         << " pho.size " << pho.size()
         ;
    assert( in_range );
    if(!in_range) std::raise(SIGINT);

    // check labels of parent and child are as expected
    const spho& parent_label = pho[idx];

    bool label_expect = label.isSameLineage( parent_label) && label.gen() == parent_label.gen() + 1  ;
    if(!label_expect) std::raise(SIGINT);
    assert( label_expect );


    // every G4Track has its own label but for reemission the lineage
    // through the generations is reflected by multiple such track labels
    // all having the same lineage (label.gs,label.ix,label.id) but different
    // reemission generations


    const sgs& _gs = get_gs(label);
    bool expected_gentype = OpticksGenstep_::IsExpected(_gs.gentype);  // SI/CK/TO
    if(!expected_gentype) std::raise(SIGINT);
    assert(expected_gentype);
    // NB: within scintillator, photons of any gentype may undergo reemission

    const sphoton& parent_photon = photon[idx] ;
    unsigned parent_idx = parent_photon.idx() ;
    bool parent_idx_expect = parent_idx == idx  ;
    assert( parent_idx_expect );
    if(!parent_idx_expect) std::raise(SIGINT);


    // replace pho[idx] and current_pho with the new generation label
    pho[idx] = label ;
    current_pho = label ;


    // RE-WRITE HISTORY : CHANGING BULK_ABSORB INTO BULK_REEMIT

    if( evt->photon )
    {
        current_ctx.p = photon[idx] ;
        sphoton& current_photon = current_ctx.p ;

        rjoinPhotonCheck(current_photon);
        current_photon.flagmask &= ~BULK_ABSORB  ; // scrub BULK_ABSORB from flagmask
        current_photon.set_flag(BULK_REEMIT) ;     // gets OR-ed into flagmask
    }

#ifndef PRODUCTION
    const int& bounce = slot[idx] ; assert( bounce > 0 );
    int prior = bounce - 1 ;
    //int num_slots = evt->max_bounce + 1  ;
    int num_slots = SEventConfig::RecordLimit()  ;
    // at truncation point and beyond cannot compare or do rejoin fixup
    if( evt->seq && prior < num_slots  )
    {
        current_ctx.seq = seq[idx] ;
        sseq& current_seq = current_ctx.seq ;

        unsigned seq_flag = current_seq.get_flag(prior);
        rjoinSeqCheck(seq_flag);
        current_seq.set_flag(prior, BULK_REEMIT);
    }

    // at truncation point and beyond cannot compare or do rejoin fixup
    if( evt->record && prior < evt->max_record )
    {
        sphoton& rjoin_record = evt->record[evt->max_record*idx+prior]  ;
        rjoinRecordCheck(rjoin_record, current_ctx.p);
        rjoin_record.flagmask &= ~BULK_ABSORB ; // scrub BULK_ABSORB from flagmask
        rjoin_record.set_flag(BULK_REEMIT) ;
    }
#endif


    // NOT: rec  (compressed record) are not handled
    //      but no longer using that as decided full recording
    //      is a debugging only activity so no point in going
    //      to great lengths squeezing memory usage for something that
    //      is just used while debugging
}



void SEvt::rjoinRecordCheck(const sphoton& rj, const sphoton& ph  ) const
{
    assert( rj.idx() == ph.idx() );
    unsigned idx = rj.idx();
    bool d12match = sphoton::digest_match( rj, ph, 12 );
    if(!d12match) dbg->d12match_fail++ ;
    if(!d12match) ComparePhotonDump(rj, ph) ;
    if(!d12match) std::cout
        << " idx " << idx
        << " slot[idx] " << slot[idx]
        << " evt.max_record " << evt->max_record
        << " d12match " << ( d12match ? "YES" : "NO" ) << std::endl
        ;
    assert( d12match );
}

void SEvt::ComparePhotonDump(const sphoton& a, const sphoton& b )  // static
{
    unsigned a_flag = a.flag() ;
    unsigned b_flag = a.flag() ;
    std::cout
        << " a.flag "  << OpticksPhoton::Flag(a_flag)
        << std::endl
        << a.desc()
        << std::endl
        << " b.flag "  << OpticksPhoton::Flag(b_flag)
        << std::endl
        << b.desc()
        << std::endl
        ;
}



/**
SEvt::rjoinPhotonCheck
------------------------

Would expect all rejoin to have BULK_ABSORB, but with
very artifical single volume of Scintillator geometry keep getting photons
hitting world boundary giving MISS.

What about perhaps reemission immediately after reemission ? Nope, I think
a new point would always be minted so the current_photon flag should
never be BULK_REEMIT when rjoin runs.

**/

void SEvt::rjoinPhotonCheck(const sphoton& ph ) const
{
    dbg->rjoinPhotonCheck++ ;

    unsigned flag = ph.flag();
    unsigned flagmask = ph.flagmask ;

    bool flag_AB = flag == BULK_ABSORB ;
    bool flag_MI = flag == MISS ;
    bool flag_xor = flag_AB ^ flag_MI ;

    if(flag_AB) dbg->rjoinPhotonCheck_flag_AB++ ;
    if(flag_MI) dbg->rjoinPhotonCheck_flag_MI++ ;
    if(flag_xor) dbg->rjoinPhotonCheck_flag_xor++ ;

    bool flagmask_AB = flagmask & BULK_ABSORB  ;
    bool flagmask_MI = flagmask & MISS ;
    bool flagmask_or = flagmask_AB | flagmask_MI ;

    if(flagmask_AB) dbg->rjoinPhotonCheck_flagmask_AB++ ;
    if(flagmask_MI) dbg->rjoinPhotonCheck_flagmask_MI++ ;
    if(flagmask_or) dbg->rjoinPhotonCheck_flagmask_or++ ;

    bool expect = flag_xor && flagmask_or ;
    LOG_IF(fatal, !expect)
        << "rjoinPhotonCheck : unexpected flag/flagmask"
        << " flag_AB " << flag_AB
        << " flag_MI " << flag_MI
        << " flag_xor " << flag_xor
        << " flagmask_AB " << flagmask_AB
        << " flagmask_MI " << flagmask_MI
        << " flagmask_or " << flagmask_or
        << ph.descFlag()
        << std::endl
        << ph.desc()
        << std::endl
        ;
    assert(expect);

}

void SEvt::rjoinSeqCheck(unsigned seq_flag) const
{
    dbg->rjoinSeqCheck++ ;

    bool flag_AB = seq_flag == BULK_ABSORB ;
    bool flag_MI = seq_flag == MISS ;
    bool flag_xor = flag_AB ^ flag_MI ;

    if(flag_AB)  dbg->rjoinSeqCheck_flag_AB++ ;
    if(flag_MI)  dbg->rjoinSeqCheck_flag_MI++ ;
    if(flag_xor) dbg->rjoinSeqCheck_flag_xor++ ;

    LOG_IF(fatal, !flag_xor) << " flag_xor FAIL " << OpticksPhoton::Abbrev(seq_flag) << std::endl ;
    assert( flag_xor );
}

/**
SEvt::pointPhoton : only used for hostside running
---------------------------------------------------

Invoked from U4Recorder::UserSteppingAction_Optical to cause the
current photon to be recorded into record vector.

The pointPhoton and finalPhoton methods need to do the hostside equivalent of
what CSGOptiX/CSGOptiX7.cu:simulate does device side, so have setup the environment to match:

ctx.point looks like it is populating sevent arrays, but actually are also populating
the vectors thanks to setting of the sevent pointers in SEvt::hostside_running_resize
so that the pointers follow around the vectors as they get reallocated.

TODO: truncation : bounce < max_bounce
at truncation ctx.point/ctx.trace stop writing anything but bounce keeps incrementing
**/

void SEvt::pointPhoton(const spho& label)
{
    dbg->pointPhoton++ ;

    assert( label.isSameLineage(current_pho) );
    unsigned idx = label.id ;
    sctx& ctx = current_ctx ;

#ifndef PRODUCTION
    t_PenultimatePoint = t_LastPoint ;
    t_LastPoint = sstamp::Now() ;
    quad4& aux = current_ctx.aux ;
    quadx4& auxx = (quadx4&)aux ;
    auxx.q3.w.x = t_LastPoint ;  // shoe-horn uint64_t time stamp into aux
#endif
    assert( ctx.idx == idx );
    int& bounce = slot[idx] ;

    bool first_point = bounce == 0 ;
    bool first_flag = ctx.p.flagmask_count() == 1 ;
    bool fake_first = first_flag == true && first_point == false ;
    LOG_IF(LEVEL, fake_first)
        << " fake_first detected, bounce: " << bounce
        << " EARLY EXIT pointPhoton"
        << " this happens when the POST of the first step is fake "
        << " resulting in GENFLAG being repeated as 2nd step first_flag still true "
        ;
    if(fake_first) return ;


#ifndef PRODUCTION
    if(first_point == false) ctx.trace(bounce);
    ctx.point(bounce);  // sseq::add_nibble the current photon flag
#endif

    LOG(LEVEL)
        << "(" << std::setw(5) << label.id
        << "," << std::setw(2) << bounce
        << ") "
        << std::setw(2) << OpticksPhoton::Abbrev(ctx.p.flag())
#ifndef PRODUCTION
        << ctx.seq.desc_seqhis()
#endif
        ;


    LOG(LEVEL)
        << " idx " << idx
        << " bounce " << bounce
        << " first_point " << first_point
        << " evt.max_record " << evt->max_record
        << " evt.max_rec    " << evt->max_rec
        << " evt.max_seq    " << evt->max_seq
        << " evt.max_prd    " << evt->max_prd
        << " evt.max_tag    " << evt->max_tag
        << " evt.max_flat    " << evt->max_flat
        << " label.desc " << label.desc()
#ifndef PRODUCTION
        << " ctx.seq.desc_seqhis " << ctx.seq.desc_seqhis() ;
#endif
        ;

    bounce += 1 ;   // increments slot array, by reference
}

/**
SEvt::addTag
-------------

HMM: this needs to be called following every random consumption ... see U4Random::flat

**/

bool SEvt::PIDX_ENABLED = ssys::getenvbool("SEvt__PIDX_ENABLED") ;

#ifndef PRODUCTION
void SEvt::addTag(unsigned tag, float flat)
{
    if(evt->tag == nullptr) return  ;
    stagr& tagr = current_ctx.tagr ;
    unsigned idx = current_ctx.idx ;

    LOG_IF(info, PIDX_ENABLED && int(idx) == PIDX )
        << " idx " << idx
        << " PIDX " << PIDX
        << " tag " << tag
        << " flat " << flat
        << " evt.tag " << evt->tag
        << " tagr.slot " << tagr.slot
        ;

    tagr.add(tag,flat)  ;


    if(random)
    {
        int flat_cursor = random->getFlatCursor();
        assert( flat_cursor > -1 );
        double flat_prior = random->getFlatPrior();
        bool cursor_slot_match = unsigned(flat_cursor) == tagr.slot ;
        LOG_IF(error, !cursor_slot_match)
            << " idx " << idx
            << " cursor_slot_match " << cursor_slot_match
            << " flat " << flat
            << " tagr.slot " << tagr.slot
            << " ( from SRandom "
            << " flat_prior " << flat_prior
            << " flat_cursor " << flat_cursor
            << "  ) "
            << std::endl
            << " MISMATCH MEANS ONE OR MORE PRIOR CONSUMPTIONS WERE NOT TAGGED "
            ;
        assert( cursor_slot_match );
    }
}

int SEvt::getTagSlot() const
{
    if(evt->tag == nullptr) return -1 ;
    const stagr& tagr = current_ctx.tagr ;
    return tagr.slot ;
}
#endif



/**
SEvt::finalPhoton : only used for hostside running
------------------------------------------------------

Canonically called from U4Recorder::PostUserTrackingAction_Optical

1. asserts label is same lineage as current_pho
2. using ProcessHits EPH enum value may change final photon step flags
   of SURFACE_DETECT(SD) to EFFICIENCY_COLLECT(EC) or EFFICIENCY_CULL(EX)

   Q: Does SD need to be scrubbed from flagmask to correspond to A side ?

3. calls sctx::end on ctx, copying ctx.seq into evt->seq[idx]
4. copies ctx.p into evt->photon[idx]

**/
void SEvt::finalPhoton(const spho& label)
{
    dbg->finalPhoton++ ;
    LOG(LEVEL) << label.desc() ;
    assert( label.isSameLineage(current_pho) );

    unsigned idx = label.id ;
    sctx& ctx = current_ctx ;
    assert( ctx.idx == idx );

#ifndef PRODUCTION
#ifdef WITH_SUP
    quadx6& xsup = (quadx6&)ctx.sup ;
    xsup.q0.w.y = sstamp::Now();
    xsup.q1.w.x = t_PenultimatePoint ;
    xsup.q1.w.y = t_LastPoint ;
#endif
    ctx.end();   // copies {seq,sup} into evt->{seq,sup}[idx] (and tag, flat when DEBUG_TAG)
#endif

    evt->photon[idx] = ctx.p ;   // HUH: why not do this in ctx.end ?
}





/**
SEvt::AddProcessHitsStamp
---------------------------

As ProcessHits may be called multiple
times for each photon this records the
timestamp range of those calls and the count.

Note that this relies on being zeroed for each photon.

Also note that the only thing specific to "ProcessHits"
is the convention of where to store the stamp range. It
is up to the user to call this from the right place.

**/

void SEvt::AddProcessHitsStamp(int idx, int p) // static
{
    if(Exists(idx)) Get(idx)->addProcessHitsStamp(p) ;
}
/**
SEvt::addProcessHitsStamp
---------------------------

See squadx.h and search for WITH_SUP to find where sup is populated.  Also see sevt.py for usage.

+--------------------------------------+----------------------------------------------------------------------------------+
| sctx::sup fields                     |                                                                                  |
+==================+===================+==================================================================================+
|  q0.w.x          |  q0.w.y           | q0.w.x SEvt::beginPhoton q0.w.y SEvt::finalPhoton                                |
+------------------+-------------------+----------------------------------------------------------------------------------+
|  q1.w.x          |  q1.w.y           | q1.w.x SEvt::finalPhoton t_PenultimatePoint q1.w.y SEvt::finalPhoton t_LastPoint |
+------------------+-------------------+----------------------------------------------------------------------------------+
|  q2.w.x          |  q2.w.y           | time range from SEvt::addProcessHitsStamp(0)                                     |
+---------+--------+--------+----------+----------------------------------------------------------------------------------+
|  q3.u.x | q3.u.y | q3.u.x | q3.u.w   | .x call count from SEvt::addProcessHitsStamp(0)                                  |
+---------+--------+--------+----------+----------------------------------------------------------------------------------+
|  q4.w.x          |  q4.w.y           | time range from SEvt::addProcessHitsStamp(1)                                     |
+---------+--------+-------------------+----------------------------------------------------------------------------------+
|  q5.u.x | q5.u.y | q5.u.z | q5.u.w   | .x call count frmo SEvt::addProcessHitsStamp(1)                                  |
+---------+--------+--------+----------+----------------------------------------------------------------------------------+

Use BP=SEvt::addProcessHitsStamp to find where time stamps are coming. Currently none.

**/
void SEvt::addProcessHitsStamp(int p)
{
    assert( p > -1 );

#ifndef PRODUCTION
#ifdef WITH_SUP
    uint64_t now = sstamp::Now();

    quad6&  sup  = current_ctx.sup ;
    quadx6& xsup = (quadx6&)current_ctx.sup ;

    uint64_t* h0 = nullptr ;
    uint64_t* h1 = nullptr ;
    unsigned* hc = nullptr ;

    switch(p)
    {
       case 0: { h0 = &xsup.q2.w.x ; h1 = &xsup.q2.w.y ; hc = &sup.q3.u.x ;} ; break ;
       case 1: { h0 = &xsup.q4.w.x ; h1 = &xsup.q4.w.y ; hc = &sup.q5.u.x ;} ; break ;
    }
    assert( hc && h0 && h1 );

    *hc += 1 ;

    if(*h0 == 0)
    {
        *h0 = now ;
    }
    else if(*h1 == 0)
    {
        *h1 = now ;
    }
    else if( *h0 > 0 && *h1 > 0 )
    {
        *h1 = now ;
    }

    /*
    std::cout
        << "SEvt::addProcessHitsStamp"
        << " p " << p
        << " now " << now
        << " h0 " << *h0
        << " h1 " << *h1
        << " hc " << *hc
        << std::endl
        ;
    */
#endif
#endif
}



void SEvt::checkPhotonLineage(const spho& label) const
{
    assert( label.isSameLineage(current_pho) );
}



////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
///////// below methods handle gathering arrays and persisting, not array content //////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////


NP* SEvt::gatherPho() const {  return NPX::ArrayFromData<int>( (int*)pho.data(), int(pho.size()), 4 ); }
NP* SEvt::gatherGS() const {   return NPX::ArrayFromData<int>( (int*)gs.data(),  int(gs.size()), 4 );  }


/**
SEvt::gatherGenstep
--------------------

As gensteps always originate on CPU its kinda silly to call access gather-ing.

**/

NP* SEvt::gatherGenstep() const { return makeGenstepArrayFromVector() ; }


quad6* SEvt::getGenstepVecData() const
{
    return genstep.size() == 0 ? nullptr : (quad6*)genstep.data();
}
int SEvt::getGenstepVecSize() const
{
    return genstep.size();
}


/**
SEvt::makeGenstepArrayFromVector (formerly misnamed getGenstepArray)
----------------------------------------------------------------------

Makes NP array from contents of genstep vector

**/


NP* SEvt::makeGenstepArrayFromVector() const
{
    return NPX::ArrayFromData<float>( (float*)genstep.data(), int(genstep.size()), 6, 4 ) ;
}

std::string SEvt::descGenstepArrayFromVector() const
{
    std::stringstream ss ;
    ss << "SEvt::descGenstepArrayFromVector genstep.size " << genstep.size() << "\n" ;
    std::string str = ss.str() ;
    return str ;
}





bool SEvt::haveGenstepVec() const { return genstep.size() > 0 ; }

/**
SEvt::gatherPhoton
--------------------

1. allocates with NP::Make
2. populates by reading from photon vector using the sevent.h pointer
   that makes things follow the on device approach

NB this means the array holds an independent copy of the vector data,
as do all(?) the gather methods.

**/

NP* SEvt::gatherPhoton() const
{
    if( evt->photon == nullptr ) return nullptr ;
    NP* p = makePhoton();
    p->read2( (float*)evt->photon );
    return p ;
}

NP* SEvt::gatherRecord() const
{
    if( evt->record == nullptr ) return nullptr ;
    NP* r = makeRecord();
    r->read2( (float*)evt->record );
    return r ;
}
NP* SEvt::gatherRec() const
{
    if( evt->rec == nullptr ) return nullptr ;
    NP* r = makeRec();
    r->read2( (short*)evt->rec );
    return r ;
}
NP* SEvt::gatherAux() const
{
    if( evt->aux == nullptr ) return nullptr ;
    NP* r = makeAux();
    r->read2( (float*)evt->aux );
    return r ;
}
NP* SEvt::gatherSup() const
{
    if( evt->sup == nullptr ) return nullptr ;
    NP* p = makeSup();
    p->read2( (float*)evt->sup );
    return p ;
}
NP* SEvt::gatherSeq() const
{
    if( evt->seq == nullptr ) return nullptr ;
    NP* s = makeSeq();
    s->read2( (unsigned long long*)evt->seq );
    return s ;
}
NP* SEvt::gatherPrd() const
{
    if( evt->prd == nullptr ) return nullptr ;
    NP* p = makePrd();
    p->read2( (float*)evt->prd );
    return p ;
}
NP* SEvt::gatherTag() const
{
    if( evt->tag == nullptr ) return nullptr ;
    NP* p = makeTag();
    p->read2( (unsigned long long*)evt->tag );
    return p ;
}
NP* SEvt::gatherFlat() const
{
    if( evt->flat == nullptr ) return nullptr ;
    NP* p = makeFlat();
    p->read2( (float*)evt->flat );
    return p ;
}
NP* SEvt::gatherSeed() const   // COULD BE IMPLEMENTED : IF NEEDED TO DEBUG  SLOT->GS ASSOCIATION
{
    LOG(fatal) << " not implemented for hostside running : getting this error indicates CompMask mixup " ;
    assert(0);
    return nullptr ;
}

/**
SEvt::gatherHit
-------------------------------------------

Does CPU side equivalent of QEvent::gatherHit_
using the photon array and the sphoton_selector

HMM: notice that this relies on having gathered
the photon array, and there being an entry in the fold.
So cannot have HitOnly on B side ?

This means that hit must come after photon in the component order

**/

NP* SEvt::gatherHit() const
{
    const NP* p = fold->get(SComp::PHOTON_) ;
    // cannot use getPhoton here as that gets the photons from topfold which is only populated after gather
    NP* h = p ? p->copy_if<float, sphoton>(*selector) : nullptr ;
    return h ;
}

NP* SEvt::gatherSimtrace() const
{
    LOG_IF(info, SIMTRACE) << " evt->simtrace " << ( evt->simtrace ? "YES" : "NO " ) ;
    if( evt->simtrace == nullptr ) return nullptr ;
    NP* p = makeSimtrace();
    p->read2( (float*)evt->simtrace );
    return p ;
}

/**
SEvt::makePhoton
-----------------

This is called by SEvt::gatherPhoton

**/

NP* SEvt::makePhoton() const
{
    NP* p = NP::Make<float>( evt->num_photon, 4, 4 );
    return p ;
}

NP* SEvt::makeRecord() const
{
    NP* r = NP::Make<float>( evt->num_photon, evt->max_record, 4, 4 );
    r->set_meta<std::string>("rpos", "4,GL_FLOAT,GL_FALSE,64,0,false" );  // eg used by examples/UseGeometryShader
    return r ;
}
NP* SEvt::makeRec() const
{
    NP* r = NP::Make<short>( evt->num_photon, evt->max_rec, 2, 4);   // stride:  sizeof(short)*2*4 = 2*2*4 = 16
    r->set_meta<std::string>("rpos", "4,GL_SHORT,GL_TRUE,16,0,false" );  // eg used by examples/UseGeometryShader
    return r ;
}
NP* SEvt::makeAux() const
{
    NP* r = NP::Make<float>( evt->num_photon, evt->max_aux, 4, 4 );
    return r ;
}
NP* SEvt::makeSup() const
{
    NP* p = NP::Make<float>( evt->num_photon, 6, 4 );
    return p ;
}

NP* SEvt::makeSeq() const
{
    return NP::Make<unsigned long long>( evt->num_seq, 2, sseq::NSEQ );
}
NP* SEvt::makePrd() const
{
    return NP::Make<float>( evt->num_photon, evt->max_prd, 2, 4);
}

/**
SEvt::makeTag
---------------

The evt->max_tag are packed into the stag::NSEQ of each *stag* struct

For example with stag::NSEQ = 2 there are two 64 bit "unsigned long long"
in the *stag* struct which with 5 bits per tag is room for 2*12 = 24 tags

**/

NP* SEvt::makeTag() const
{
    assert( sizeof(stag) == sizeof(unsigned long long)*stag::NSEQ );
    return NP::Make<unsigned long long>( evt->num_photon, stag::NSEQ);   //
}
NP* SEvt::makeFlat() const
{
    assert( sizeof(sflat) == sizeof(float)*sflat::SLOTS );
    return NP::Make<float>( evt->num_photon, sflat::SLOTS );   //
}
NP* SEvt::makeSimtrace() const
{
    return NP::Make<float>( evt->num_simtrace, 4, 4 );
}






/**
SEvt::getMeta (an SCompProvider method)
-----------------------------------------

NB as metadata is collected in SEvt::gather_components
from the SCompProvider this is NOT the source of metadata
for GPU running, that comes from QEvent::getMeta.
This is only the source for CPU U4Recorder running (aka B evts).

**/


std::string SEvt::getMeta() const
{
    return meta ;
}

const char* SEvt::getTypeName() const
{
    return TYPENAME ;
}




/**
SEvt::gatherComponent
------------------------

If the cmp is within the SEventConfig::GatherComp mask
then invoke the corresponding gather... method otherwise
return nullptr.

NB this is for hostside running only, for device-side running
the SCompProvider is the QEvent not the SEvt, so this method
is not called

**/

NP* SEvt::gatherComponent(unsigned cmp) const
{
    unsigned gather_mask = SEventConfig::GatherComp();
    return gather_mask & cmp ? gatherComponent_(cmp) : nullptr ;
}

NP* SEvt::gatherComponent_(unsigned cmp) const
{
    NP* a = nullptr ;
    switch(cmp)
    {
        case SCOMP_INPHOTON:  a = gatherInputPhoton() ; break ;
        case SCOMP_G4STATE:   a = gatherG4State()     ; break ;

        case SCOMP_GENSTEP:   a = gatherGenstep()  ; break ;
        case SCOMP_DOMAIN:    a = gatherDomain()   ; break ;
        case SCOMP_PHOTON:    a = gatherPhoton()   ; break ;
        case SCOMP_RECORD:    a = gatherRecord()   ; break ;
        case SCOMP_REC:       a = gatherRec()      ; break ;
        case SCOMP_AUX:       a = gatherAux()      ; break ;
        case SCOMP_SUP:       a = gatherSup()      ; break ;
        case SCOMP_SEQ:       a = gatherSeq()      ; break ;
        case SCOMP_PRD:       a = gatherPrd()      ; break ;
        case SCOMP_TAG:       a = gatherTag()      ; break ;
        case SCOMP_FLAT:      a = gatherFlat()     ; break ;

        case SCOMP_SEED:      a = gatherSeed()     ; break ;
        case SCOMP_HIT:       a = gatherHit()      ; break ;
        case SCOMP_SIMTRACE:  a = gatherSimtrace() ; break ;
        case SCOMP_PHO:       a = gatherPho()      ; break ;
        case SCOMP_GS:        a = gatherGS()       ; break ;
    }
    return a ;
}


/**
SEvt::saveGenstep
-----------------

The returned array takes a full copy of the genstep quad6 vector
with all gensteps collected since the last SEvt::clear.
The array is thus independent from quad6 vector, and hence is untouched
by SEvt::clear

**/

void SEvt::saveGenstep(const char* dir) const  // HMM: NOT THE STANDARD SAVE
{
    NP* a = gatherGenstep();
    if(a == nullptr) return ;
    LOG(LEVEL) << a->sstr() << " dir " << dir ;
    a->save(dir, "gs.npy");
}
void SEvt::saveGenstepLabels(const char* dir, const char* name) const
{
    NP::Write<int>(dir, name, (int*)gs.data(), gs.size(), 4 );
}

std::string SEvt::descGS() const
{
    std::stringstream ss ;
    for(unsigned i=0 ; i < getNumGenstepFromGenstep() ; i++) ss << gs[i].desc() << std::endl ;
    std::string s = ss.str();
    return s ;
}

std::string SEvt::descDir() const
{
    const char* savedir = getSaveDir();
    const char* loaddir = getLoadDir();
    std::stringstream ss ;
    ss
       << " savedir " << ( savedir ? savedir : "-" )
       << std::endl
       << " loaddir " << ( loaddir ? loaddir : "-" )
       << std::endl
       << " is_loaded " << ( is_loaded ? "YES" : "NO" )
       << std::endl
       << " is_loadfail " << ( is_loadfail ? "YES" : "NO" )
       << std::endl
       ;
    std::string s = ss.str();
    return s ;
}

std::string SEvt::descFold() const
{
    return topfold->desc();
}
std::string SEvt::Brief() // static
{
    std::stringstream ss ;
    ss << "SEvt::Brief "
       << " SEvt::Exists(0) " << ( Exists(0) ? "Y" : "N" )
       << " SEvt::Exists(1) " << ( Exists(1) ? "Y" : "N" )
       << std::endl
       << " SEvt::Get(0)->brief() " << ( Exists(0) ? Get(0)->brief() : "-" )
       << std::endl
       << " SEvt::Get(1)->brief() " << ( Exists(0) ? Get(0)->brief() : "-" )
       << std::endl
       ;
    std::string str = ss.str();
    return str ;
}
std::string SEvt::brief() const
{
    std::stringstream ss ;
    ss << "SEvt::brief "
       << " getIndex " << getIndex()
       << " hasInputPhoton " << ( hasInputPhoton() ? "Y" : "N" )
       << " hasInputPhotonTransformed " << ( hasInputPhotonTransformed() ? "Y" : "N" )
       ;
    std::string str = ss.str();
    return str ;
}

std::string SEvt::id() const
{
    bool is_egpu = isEGPU();
    bool is_ecpu = isECPU();

    std::stringstream ss ;
    ss << "SEvt::id "
       << ( is_egpu ? "EGPU" : "" )
       << ( is_ecpu ? "ECPU" : "" )
       << " (" << getIndexPresentation() << ") "
       << " GSV " << ( haveGenstepVec() ? "YES" : "NO " )
       << " " << descStage()
       ;
    std::string str = ss.str();
    return str ;
}





std::string SEvt::desc() const
{
    std::stringstream ss ;
    ss << evt->desc()
       << std::endl
       << descDir()
       << std::endl
       << dbg->desc()
       << std::endl
       << " g4state " << ( g4state ? g4state->sstr() : "-" )
       << std::endl
       << " SEventConfig::Initialize_COUNT " << SEventConfig::Initialize_COUNT
       << std::endl
       ;
    std::string str = ss.str();
    return str ;
}

std::string SEvt::descDbg() const
{
    std::stringstream ss ;
    ss << dbg->desc() << std::endl ;
    std::string str = ss.str();
    return str ;
}

/**
SEvt::gather_components : collects fresh arrays into NPFold from provider
---------------------------------------------------------------------------

SEvt::gather_components is invoked by SEvt::gather from QSim::simulate::


     +-------------------+                 +-----------------+
     | QEvent/GPU buf    |                 |  SEvt/NPFold    |
     |   OR              | === gather ===> |                 |
     | SEvt vecs         |                 |                 |
     +-------------------+                 +-----------------+



The SEvt::gather was formerly invoked from SEvt::save, but that
is incompatible with getting access to hits so the invokation
of the gather must now be done from high level methods such
as QSim::simulate.


0. invokes NPFold::add_subfold on the *topfold* giving *fold*
   into which the gathering is done. This is done to support
   multilaunch running by invoking NPFold::concat on the *topfold*
   after the launch loop of QSim::simulate

1. invokes gatherComponent on the SCompProvider instance which is either
   this SEvt instance for CPU/U4Recorder running OR the QEvent instance
   for GPU/QSim runnning

   * the SCompProvider allocates an NP array and populates it either
     from vectors for CPU running or by copies from GPU device buffers

2. the freshly created NP arrays are added to the NPFold,
   NB pre-existing keys cause NPFold asserts, so it is essential
   that SEvt::clear is called to clear the fold before gathering

Note thet QEvent::setGenstep invoked SEvt::clear so the genstep vectors
are clear when this gets called. So must rely on the contents of the
fold to get the stats.

**/

void SEvt::gather_components()   // *GATHER*
{
    fold = topfold->add_subfold();
    if(NPFOLD_VERBOSE) fold->set_verbose();
    const char* fkey = topfold->get_last_subfold_key();  // f000 f001 f001 ...


    int num_genstep = -1 ;
    int num_photon  = -1 ;
    int num_hit     = -1 ;

    int num_comp = gather_comp.size() ;

    LOG(LEVEL) << " num_comp " << num_comp << " from provider " << provider->getTypeName() << " fkey " << ( fkey ? fkey : "-" )   ;
    LOG_IF(info, GATHER||SIMTRACE) << " num_comp " << num_comp << " from provider " << provider->getTypeName() << " fkey " << ( fkey ? fkey : "-" ) ;

    for(int i=0 ; i < num_comp ; i++)
    {
        unsigned cmp = gather_comp[i] ;
        const char* k = SComp::Name(cmp);
        NP* a = provider->gatherComponent(cmp);  // see QEvent::gatherComponent for GPU running
        bool null_component = a == nullptr ;

        LOG(LEVEL)
            << " k " << std::setw(15) << k
            << " a " << ( a ? a->brief() : "-" )
            << " null_component " << ( null_component ? "YES" : "NO " )
            ;

        LOG_IF(info, GATHER)
            << " k " << std::setw(15) << k
            << " a " << ( a ? a->brief() : "-" )
            << " null_component " << ( null_component ? "YES" : "NO " )
            ;


        if(null_component) continue ;
        fold->add(k, a);

        int num = a->shape[0] ;
        if(     SComp::IsGenstep(cmp)) num_genstep = num ;
        else if(SComp::IsPhoton(cmp))  num_photon = num ;
        else if(SComp::IsHit(cmp))     num_hit = num ;
    }

    gather_total += 1 ;

    if(num_genstep > -1) genstep_total += num_genstep ;
    if(num_photon > -1)  photon_total += num_photon ;
    if(num_hit > -1)     hit_total += num_hit ;

    LOG(LEVEL)
        << " num_comp " << num_comp
        << " num_genstep " << num_genstep
        << " num_photon " << num_photon
        << " num_hit " << num_hit
        << " gather_total " << gather_total
        << " genstep_total " << genstep_total
        << " photon_total " << photon_total
        << " hit_total " << hit_total
        ;
}


/**
SEvt::gather_metadata
----------------------

Lots of timing metadata, so try calling from SEvt::endOfEvent not SEvt::gather

HMM: replaces fold.meta with metadata from provider : either this SEvt or QEvent ?

**/


void SEvt::gather_metadata()
{
    if(topfold == nullptr)
    {
        LOG_IF(error, gather_metadata_notopfold < 10) << " gather_metadata_notopfold " << gather_metadata_notopfold ;
        gather_metadata_notopfold += 1 ;
        return ;
    }

    std::string provmeta = provider->getMeta();
    LOG(LEVEL) << " provmeta ["<< provmeta << "]" ;
    topfold->meta = provmeta ;
}



/**
SEvt::gather
-------------

Collects the components configured by SEventConfig::CompMask
into NPFold from the SCompProvider which can either be:

* this SEvt instance for hostside running
* the qudarap/QEvent instance for deviceside running, eg G4CXSimulateTest


For multi-launch running genstep slices are uploaded
before each launch and *gather* is called after each launch.


For an example of where this should be invoked from see QSim::simulate

**/

void SEvt::gather()
{
    setStage(SEvt__gather);
    LOG_IF(info, LIFECYCLE) << id() ;

    gather_components();
}


/**
SEvt::add_array
-----------------

Q: WHO CALLS THIS ?
A: QSim::fake_propagate

**/


void SEvt::add_array( const char* k, const NP* a )
{
    LOG(LEVEL) << " k " << k << " a " << ( a ? a->sstr() : "-" ) ;
    extrafold->add(k, a);
}

void SEvt::addEventConfigArray()
{
    extrafold->add(SEventConfig::NAME, SEventConfig::Serialize() );
}

/**
SEvt::addProcessHits_EPH
-------------------------

Called from U4Recorder::addProcessHits_EPH but as that is
called from U4RecorderAnaMgr::EndOfEventAction it is too late
for the extrafold save.

**/

void SEvt::addProcessHits_EPH(NP* eph_meta)
{
    LOG_IF(info, EPH_) << " eph_meta " << ( eph_meta ? eph_meta->sstr() : "-" ) ;

    saveExtra( "SProcessHits_EPH.npy", eph_meta );
}




int SEvt::load()
{
    const char* base = DefaultBase();
    int rc = load(base);
    LOG(LEVEL) << "SEvt::DefaultBase " << base << " rc " << rc ;
    return rc ;
}


bool SEvt::hasIndex() const
{
    return index != MISSING_INDEX ;
}

bool SEvt::hasInstance() const
{
    return instance != MISSING_INSTANCE ;
}


/**
SEvt::DefaultBase
-------------------

If argument is defined it is used, otherwise::

   $TMP/GEOM/$GEOM/$ExecutableName

Note that when Opticks is used from python the OPTICKS_SCRIPT
envvar can be used to override the not very useful detected
ExecutableName of "python3.11" or whatever.

**/

const char* SEvt::DefaultBase(const char* base_) // static
{
    const char* base = base_ ? base_ : spath::DefaultOutputDir() ;
    return base ;
}


/**
SEvt::RunDir
--------------

Directory without event index, used for run level metadata.

**/

const char* SEvt::RunDir( const char* base_ )  // static
{
    const char* base = DefaultBase(base_);
    const char* reldir = SEventConfig::EventReldir() ;
    const char* dir = spath::Resolve(base, reldir );

    bool is_save_nothing = IsSaveNothing();
    if(!is_save_nothing) sdirectory::MakeDirs(dir,0);
    return dir ;
}

/**
SEvt::getDir
----------------------------

Reimpl in a way that is faster to understand,
in attempt to remove lots of tedious code by
focussing tokenization into spath.h and the
high level control here in one place.

HMM: could expand on that approach exposing ALL$VERSION
here too instead of hiding in Reldir

Example with::

    TMP               /data/blyth/opticks
    GEOM              J_2024nov27
    ExecutableName    CSGOptiXSMTest
    VERSION           1
    Reldir            ALL1
    sidx:IndexString  A000
    OutputDir         /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1/A000/

**/

const char* SEvt::getDir(const char* base_) const
{
    const char* base = DefaultBase(base_);
    const char* reldir = SEventConfig::EventReldir() ;
    const char* sidx = hasIndex() ? getIndexString(nullptr) : nullptr ;
    const char* path = sidx ? spath::Resolve(base,reldir,sidx ) : spath::Resolve(base, reldir) ;

    bool is_save_nothing = IsSaveNothing();
    if(!is_save_nothing) sdirectory::MakeDirs(path,0);

    LOG_IF(info, DIRECTORY || SIMTRACE || LIFECYCLE )
        << std::endl
        << " base_  " << ( base_ ? base_ : "-" ) << "\n"
        << " SEventConfig::EventReldir   " << ( reldir ? reldir : "-" ) << "\n"
        << " SEventConfig::_EventReldirDefault " << SEventConfig::_EventReldirDefault << "\n"
        << " sidx   " << ( sidx ? sidx : "-" ) << "\n"
        << " path   " << ( path ? path : "-" ) << "\n"
        ;

    return path ;
}

char SEvt::getInstancePrefix() const
{
    char pfx = '\0' ;
    switch(instance)
    {
       case EGPU:             pfx = 'A' ; break ;
       case ECPU:             pfx = 'B' ; break ;
       case MISSING_INSTANCE: pfx = 'M' ; break ;
       default:               pfx = 'D' ; break ;
    }
    return pfx ;
}

std::string SEvt::getIndexString_(const char* hdr) const
{
    assert( index >= 0 && index != MISSING_INDEX );
    int wid = 3 ;
    char pfx = getInstancePrefix();
    return sstr::FormatIndex_(index, pfx, wid, hdr );
}

const char* SEvt::getIndexString(const char* hdr) const
{
    std::string str = getIndexString_(hdr);
    return strdup(str.c_str());
}






std::string SEvt::descSaveDir(const char* dir_) const
{
    const char* dir = getDir(dir_);
    const char* reldir = SEventConfig::EventReldir() ;
    bool with_index = index != MISSING_INDEX ;
    std::stringstream ss ;
    ss << "SEvt::descSaveDir"
       << " dir_ " << ( dir_ ? dir_ : "-" )
       << " dir  " << ( dir  ? dir  : "-" )
       << " reldir " << ( reldir ? reldir : "-" )
       << " SEventConfig::_EventReldirDefault " << SEventConfig::_EventReldirDefault
       << " with_index " << ( with_index ? "Y" : "N" )
       << " index " << ( with_index ? index : -1 )
       << " this " << std::hex << this << std::dec
       << std::endl
       ;
    std::string str = ss.str();
    return str ;
}

int SEvt::load(const char* base_)
{
    const char* dir = getDir(base_);
    LOG(LEVEL)
        << " base_ " << ( base_ ? base_ : "-" )
        << " dir " << ( dir ? dir : "-" )
        ;
    LOG_IF(fatal, dir == nullptr) << " null dir : probably missing environment : run script, not executable directly " ;
    assert(dir);
    int rc = loadfold(dir);
    return rc ;
}

int SEvt::loadfold( const char* dir )
{
    LOG(LEVEL) << "[ topfold.load " << dir ;
    int rc = topfold->load(dir);
    LOG(LEVEL) << "] topfold.load " << dir ;
    is_loaded = true ;
    onload();
    return rc ;
}


void SEvt::onload()
{
    const NP* domain = topfold->get(SComp::Name(SCOMP_DOMAIN)) ;
    if(!domain) return ;

    index = domain->get_meta<int>("index");
    instance = domain->get_meta<int>("instance");

    if(hasInstance()) // ie not MISSING_INSTANCE
    {
        Set(instance, this);
    }

    LOG(LEVEL)
        << " from domain "
        << " index " << index
        << " instance " << instance
        ;
}



void SEvt::save(const char* bas, const char* rel1, const char* rel2 )
{
    const char* dir = spath::Resolve(bas, rel1, rel2);
    save(dir);
}
void SEvt::save(const char* bas, const char* rel )
{
    const char* dir = spath::Resolve(bas, rel);
    save(dir);
}

/**
SEvt::save
--------------

The component arrays are gathered by SEvt::gather_components
into the NPFold and then saved. Which components to gather and save
are configured via SEventConfig::GatherComp and SEventConfig::SaveComp
using the SComp enumeration.

The arrays are gathered from the SCompProvider object, which
may be QEvent for on device running or SEvt itself for U4Recorder
Geant4 tests.

SEvt::save persists NP arrays into the default directory
or the directory argument provided.

The FALLBACK_DIR which is used for the SEvt::DefaultDir is obtained from SEventConfig::OutFold
which is normally "$DefaultOutputDir" $TMP/GEOM/$GEOM/ExecutableName
This can be overriden using SEventConfig::SetOutFold or by setting the
envvar OPTICKS_OUT_FOLD.

It is normally much easier to use the default of "$DefaultOutputDir" as this
takes care of lots of the bookkeeping automatically.
However in some circumstances such as with the B side of aligned running (U4RecorderTest)
it is appropriate to use the override code or envvar to locate B side outputs together
with the A side.


**Override with TMP envvar rather than OPTICKS_OUT_FOLD to still have auto-bookkeeping**

Note that when needing to override the default output directory it is usually
preferable to use TMP envvar as most of the automatic bookkeeping will still be done in that case.

The below examples are with GEOM envvar set to "Pasta" and "FewPMT" with different executables:

+--------------------------------------------+-----------------------------------------------------------+
|   TMP envvar                               |  SEvt saveDir                                             |
+============================================+===========================================================+
|    undefined                               |   /tmp/blyth/opticks/GEOM/Pasta/SEvtTest/ALL              |
+--------------------------------------------+-----------------------------------------------------------+
|   /tmp/$USER/opticks                       |   /tmp/blyth/opticks/GEOM/Pasta/SEvtTest/ALL              |
+--------------------------------------------+-----------------------------------------------------------+
|   undefined                                |   /tmp/blyth/opticks/GEOM/FewPMT/U4SimulateTest/ALL0/000  |
+--------------------------------------------+-----------------------------------------------------------+

Only when more control of the output is needed is it appropriate to use OPTICKS_OUT_FOLD envvar.

+--------------------------------------------+-----------------------------------------------------------+
|  OPTICKS_OUT_FOLD envvar                   |  SEvt saveDir                                             |
+============================================+===========================================================+
|   undefined                                |   /tmp/blyth/opticks/GEOM/Pasta/SEvtTest/ALL              |
+--------------------------------------------+-----------------------------------------------------------+
|   /tmp/$USER/opticks                       |   /tmp/blyth/opticks/ALL                                  |
+--------------------------------------------+-----------------------------------------------------------+
|   /tmp/$USER/opticks/GEOM/$GEOM/SEvtTest   |   /tmp/blyth/opticks/GEOM/Pasta/SEvtTest/ALL              |
+--------------------------------------------+-----------------------------------------------------------+

* see tests/SEvtTest_saveDir.sh

**/

void SEvt::save()
{
    const char* base = DefaultBase();
    LOG_IF(info, LIFECYCLE || SIMTRACE) << " base [" << ( base ? base : "-" ) << "]" ;
    save(base);
}




/**
SEvt::save
-------------

Saving arrays is a debugging activity configured
with SEventConfig::DescSaveComp that has a large impact on performance.

Gathering arrays (eg downloading them from device buffers to host)
was formerly invoked from here, but the *gather* has now been moved upwards
to QSim::simulate to allow collecting hits into other collections.
The arrays to gather are configured with SEventConfig::DescGatherComp
separately from the arrays to save. Clearly its necessary to gather
a component in order to save it.

If an index has been set with SEvt::setIndex SEvt::SetIndex
and not unset with SEvt::UnsetIndex SEvt::unsetIndex
then the directory is suffixed with the index::

    /some/directory/A001


Notice the mixed memory handling in the below.
The save_fold is shallow copied from topfold, indicating
it does not own its arrays and should not be deleted.

All fine so far. But then seqnib and seqnib_table are created
and added ... so those would leak without the manual deletes
at the tail.

How to avoid the need for such care ? NPFold has a skipdelete flag,
but that is at fold level./ Perhaps need each array to have skipdelete ?

**/

void SEvt::save(const char* dir_)
{
    LOG_IF(info, LIFECYCLE || SIMTRACE || SAVE) << id() << " dir_[" << ( dir_ ? dir_ : "-" ) << "]" ;

    //  gather();   MOVED gather upwards to allow copying hits into other collections

    LOG(LEVEL) << descComponent() ;
    LOG(LEVEL) << descFold() ;

    std::string save_comp = SEventConfig::DescSaveComp() ;
    NPFold* save_fold = topfold->shallowcopy(save_comp.c_str());

    LOG_IF(info, SAVE) << " save_comp " << save_comp ;
    //LOG_IF(info, SAVE) << " topfold.desc   " << ( topfold ? topfold->desc() : "-" )  ;
    //LOG_IF(info, SAVE) << " save_fold.desc " << ( save_fold ? save_fold->desc() : "-" ) ;


    LOG_IF(LEVEL, save_fold == nullptr) << " NOTHING TO SAVE SEventConfig::SaveCompLabel/OPTICKS_SAVE_COMP  " << save_comp ;
    if(save_fold == nullptr) return ;

    const NP* seq = save_fold->get("seq");
    NP* seqnib = nullptr ;
    NP* seqnib_table = nullptr ;
    if(seq)
    {
        seqnib = CountNibbles(seq) ;
        seqnib_table = CountNibbles_Table(seqnib) ;
        save_fold->add("seqnib", seqnib );
        save_fold->add("seqnib_table", seqnib_table );
        // NPFold::add does nothing with nullptr array
    }

    if(extrafold)
    {
        int extra_items = extrafold->num_items();
        LOG_IF(info, EPH_) << "adding extra_items " << extra_items << " to save_fold " ;

        for(int i=0 ; i < extra_items ; i++)
        {
            const char* key = extrafold->get_key(i);
            const NP* arr = extrafold->get_array(i);
            save_fold->add(key, arr);
        }
    }


    int slic = save_fold->_save_local_item_count();
    if( slic > 0 )
    {
        const char* dir = getDir(dir_);   // THIS CREATES DIRECTORY
        LOG_IF(info, MINIMAL||SIMTRACE) << dir << " [" << save_comp << "]"  ;
        LOG(LEVEL) << descSaveDir(dir_) ;

        LOG(LEVEL) << "[ save_fold.save " << dir ;
        save_fold->save(dir);
        LOG(LEVEL) << "] save_fold.save " << dir ;

        int num_save_comp = SEventConfig::NumSaveComp();
        if(num_save_comp > 0 ) saveFrame(dir);
        // could add frame to the fold ?
        // for now just restrict to saving frame when other components are saved
    }
    else
    {
        LOG(LEVEL) << "SKIP SAVE AS NPFold::_save_local_item_count zero " ;
    }

    // NB: NOT DELETING save_fold AS IT IS A SHALLOW COPY : IT DOES NOT OWN THE ARRAYS
    delete seqnib ;
    delete seqnib_table ;
}















/**
SEvt::saveExtra(name,a)
------------------------

Save extra array into the standard SEvt folder
The name is expected to end with ".npy"

For example with name of "somearray.npy" might
save to::

    /data1/blyth/tmp/GEOM/SEVT_TEST/SEvtTest/ALL0_no_opticks_event_name/somearray.npy

The directory depends on the name of the executable eg "SEvtTest" and envvars::

    TMP                  /data1/blyth/tmp
    GEOM                 SEVT_TEST
    VERSION              0
    OPTICKS_EVENT_NAME   no_opticks_event_name

**/

void SEvt::saveExtra( const char* name, const NP* a  ) const
{
    const char* dir = getDir();
    LOG_IF(info, LIFECYCLE)
        << "saveExtra(name,a)\n"
        << "name[" << name << "]\n"
        << "dir [" << dir << "]\n"
        << "   a[" << ( a ? a->sstr() : "-" ) << "]"
        ;

    a->save(dir, name);
}


/**
SEvt::saveExtra(base, name,a)
---------------------------------

The name is expected to end with ".npy"

Save extra array into directory with the specified base folder,
but with the organizational reldir.  For example with base "/tmp"
and name "somearray.npy" and no other envvars or SEvt index
setup might save to::

    /tmp/ALL0_no_opticks_event_name/somearray.npy

Setting envvars VERSION and OPTICKS_EVENT_NAME would
change the reldir. Also setting an index on the event would
add a subfold like "A000" or "B000".

**/

void SEvt::saveExtra(const char* base, const char* name, const NP* a ) const
{
    const char* dir = getDir(base);
    LOG_IF(info, LIFECYCLE)
        << "saveExtra(base,name,a)\n"
        << "base[" << base << "]\n"
        << "dir [" << dir  << "]\n"
        << "name[" << name << "]\n"
        << "   a[" << ( a ? a->sstr() : "-" ) << "]"
        ;
    a->save(dir, name );
}


void SEvt::saveFrame(const char* dir) const
{
    LOG(LEVEL) << "[ dir " << dir ;
    frame.save(dir);
    LOG(LEVEL) << "] dir " << dir ;
}


std::string SEvt::descComponent() const
{
    return DescComponent(topfold);
}

std::string SEvt::DescComponent(const NPFold* f)  // static
{
    const NP* genstep  = f->get(SComp::Name(SCOMP_GENSTEP)) ;
    const NP* seed     = f->get(SComp::Name(SCOMP_SEED)) ;
    const NP* photon   = f->get(SComp::Name(SCOMP_PHOTON)) ;
    const NP* hit      = f->get(SComp::Name(SCOMP_HIT)) ;
    const NP* record   = f->get(SComp::Name(SCOMP_RECORD)) ;
    const NP* rec      = f->get(SComp::Name(SCOMP_REC)) ;
    const NP* aux      = f->get(SComp::Name(SCOMP_REC)) ;
    const NP* sup      = f->get(SComp::Name(SCOMP_SUP)) ;
    const NP* seq      = f->get(SComp::Name(SCOMP_SEQ)) ;
    const NP* domain   = f->get(SComp::Name(SCOMP_DOMAIN)) ;
    const NP* simtrace = f->get(SComp::Name(SCOMP_SIMTRACE)) ;
    const NP* g4state  = f->get(SComp::Name(SCOMP_G4STATE)) ;
    const NP* pho      = f->get(SComp::Name(SCOMP_PHO)) ;
    const NP* gs       = f->get(SComp::Name(SCOMP_GS)) ;

    std::stringstream ss ;
    ss << "SEvt::DescComponent"
       << std::endl
       << std::setw(20) << " SEventConfig::DescGatherComp " << SEventConfig::DescGatherComp() << std::endl
       << std::endl
       << std::setw(20) << " SEventConfig::DescSaveComp " << SEventConfig::DescSaveComp() << std::endl
       << std::setw(20) << "hit" << " "
       << std::setw(20) << ( hit ? hit->sstr() : "-" )
       << " "
       << std::endl
       << std::setw(20) << "seed" << " "
       << std::setw(20) << ( seed ? seed->sstr() : "-" )
       << " "
       << std::endl
       << std::setw(20) << "genstep" << " "
       << std::setw(20) << ( genstep ? genstep->sstr() : "-" )
       << " "
       << std::setw(30) << "SEventConfig::MaxGenstep"
       << std::setw(20) << SEventConfig::MaxGenstep()
       << std::endl
       << std::setw(20) << "photon" << " "
       << std::setw(20) << ( photon ? photon->sstr() : "-" )
       << " "
       << std::setw(30) << "SEventConfig::MaxPhoton"
       << std::setw(20) << SEventConfig::MaxPhoton()
       << std::endl
       << std::setw(20) << "record" << " "
       << std::setw(20) << ( record ? record->sstr() : "-" )
       << " "
       << std::setw(30) << "SEventConfig::MaxRecord"
       << std::setw(20) << SEventConfig::MaxRecord()
       << std::endl
       << std::setw(20) << "aux" << " "
       << std::setw(20) << ( aux ? aux->sstr() : "-" )
       << " "
       << std::setw(30) << "SEventConfig::MaxAux"
       << std::setw(20) << SEventConfig::MaxAux()
       << std::endl
       << std::setw(20) << "sup" << " "
       << std::setw(20) << ( sup ? sup->sstr() : "-" )
       << " "
       << std::setw(30) << "SEventConfig::MaxSup"
       << std::setw(20) << SEventConfig::MaxSup()
       << std::endl
       << std::setw(20) << "rec" << " "
       << std::setw(20) << ( rec ? rec->sstr() : "-" )
       << " "
       << std::setw(30) << "SEventConfig::MaxRec"
       << std::setw(20) << SEventConfig::MaxRec()
       << std::endl
       << std::setw(20) << "seq" << " "
       << std::setw(20) << ( seq ? seq->sstr() : "-" )
       << " "
       << std::setw(30) << "SEventConfig::MaxSeq"
       << std::setw(20) << SEventConfig::MaxSeq()
       << std::endl
       << std::setw(20) << "domain" << " "
       << std::setw(20) << ( domain ? domain->sstr() : "-" )
       << " "
       << std::endl
       << std::setw(20) << "simtrace" << " "
       << std::setw(20) << ( simtrace ? simtrace->sstr() : "-" )
       << " "
       << std::endl
       << std::setw(20) << "g4state" << " "
       << std::setw(20) << ( g4state ? g4state->sstr() : "-" )
       << " "
       << std::endl
       << std::setw(20) << "pho" << " "
       << std::setw(20) << ( pho ? pho->sstr() : "-" )
       << " "
       << std::endl
       << std::setw(20) << "gs" << " "
       << std::setw(20) << ( gs ? gs->sstr() : "-" )
       << " "
       << std::endl
       ;
    std::string str = ss.str();
    return str ;
}
std::string SEvt::descComp() const
{
    std::stringstream ss ;
    ss << "SEvt::descComp "
       << " gather_comp.size " << gather_comp.size()
       << " SComp::Desc(gather_comp) " << SComp::Desc(gather_comp)
       << std::endl
       << " save_comp.size "   << save_comp.size()
       << " SComp::Desc(save_comp) " << SComp::Desc(save_comp)
       << std::endl
       ;
    std::string s = ss.str();
    return s ;
}

std::string SEvt::descVec() const
{
    std::stringstream ss ;
    ss << "SEvt::descVec "
       << " gather_comp " << gather_comp.size()
       << " save_comp " << save_comp.size()
       << " genstep " << genstep.size()
       << " gs " << gs.size()
       << " pho " << pho.size()
       << " slot " << slot.size()
       << " photon " << photon.size()
       << " record " << record.size()
       << " rec " << rec.size()
       << " seq " << seq.size()
       << " prd " << prd.size()
       << " tag " << tag.size()
       << " flat " << flat.size()
       << " simtrace " << simtrace.size()
       << " aux " << aux.size()
       << " sup " << sup.size()
       ;
    std::string s = ss.str();
    return s ;
}



const NP* SEvt::getGenstep() const { return topfold->get(SComp::GENSTEP_) ;}
const NP* SEvt::getPhoton() const {  return topfold->get(SComp::PHOTON_) ; }
const NP* SEvt::getHit() const {     return topfold->get(SComp::HIT_) ; }
const NP* SEvt::getAux() const {     return topfold->get(SComp::AUX_) ; }
const NP* SEvt::getSup() const {     return topfold->get(SComp::SUP_) ; }
const NP* SEvt::getPho() const {     return topfold->get(SComp::PHO_) ; }
const NP* SEvt::getGS() const {      return topfold->get(SComp::GS_) ; }

unsigned SEvt::getNumPhoton() const { return topfold->get_num(SComp::PHOTON_) ; }
unsigned SEvt::getNumHit() const
{
    int num = topfold->get_num(SComp::HIT_) ;  // number of items in array
    return num == NPFold::UNDEF ? 0 : num ;   // avoid returning -1 when no hits
}

std::string SEvt::descSimulate() const
{
    unsigned num_hit = getNumHit() ;
    bool is_undef = num_hit == SEvt::UNDEF ;

    std::stringstream ss ;
    ss << "SEvt::descSimulate"
       << " instance " << getInstance()
       << " index " << getIndex()
       << " num_genstep " << getNumGenstepFromGenstep()
       << " num_photon " << getNumPhotonCollected()
       << " num_hit " << getNumHit()
       << " num_hit.is_undef " << ( is_undef ? "YES" : "NO " )
       << " sev.brief " << brief()
       ;

    std::string str = ss.str();
    return str ;
}











void SEvt::getPhoton(sphoton& p, unsigned idx) const  // global
{
    const NP* photon = getPhoton();
    sphoton::Get(p, photon, idx );
}
void SEvt::getHit(sphoton& p, unsigned idx) const
{
    const NP* hit = getHit();
    sphoton::Get(p, hit, idx );
}

/**
SEvt::getLocalPhoton
--------------------

sphoton::iindex instance index used to get instance frame
from (SGeo*)cf which is used to transform the photon

**/

void SEvt::getLocalPhoton(sphoton& lp, unsigned idx) const
{
    getPhoton(lp, idx);

    sframe fr ;
    getPhotonFrame(fr, lp);   // HMM: this is just using lp.iindex
    fr.transform_w2m(lp);
}

/**
SEvt::getLocalHit_LEAKY
------------------------

This impl was formerly SEvt::getLocalHit

1. copy *idx* hit from NP array into sphoton& lp struct
2. uses lp.iindex (instance index) to lookup the frame from the SGeo* cf geometry

   * TODO: check sensor_identifier, it should now be done GPU side already ?

Dec 19,2023 : Added sensor_identifier subtract one
to correspond to the addition of one in::

   CSGFoundry::addInstance firstcall:true
   CSGFoundry::addInstanceVector

The need for the offset by one arises from the optixInstance identifier
range limitation meaning that need zero to mean not-a-sensor
and not -1 0xffffffff

BUT: the limitation is on the GPU side geometry optixInstance identifier,
the limitation does not apply to::

1. photon/hit struct in GPU or CPU
2. stree inst/iinst that only exist CPU side

Actually the reason for not needing -1 in the new impl
is simpler than that.

It is only the sqat4.h identity that is incremented, the source identity
from the stree.h/iinst is never incremented (as never uploaded).

**/

void SEvt::getLocalHit_LEAKY(sphit& ht, sphoton& lp, unsigned idx) const
{
    getHit(lp, idx);   // copy *idx* hit from NP array into sphoton& lp struct

    sframe fr = {} ;

    getPhotonFrame(fr, lp);
    fr.transform_w2m(lp);

    ht.iindex = fr.inst() ;
    ht.sensor_identifier = fr.sensor_identifier() - 1 ;
    ht.sensor_index = fr.sensor_index();
}

/**
SEvt::getLocalHit
-------------------

Canonical usage from U4HitGet::FromEvt

This implementation gets the w2m instance transform
directly using stree::get_iinst avoiding the use of the sframe.h

The advantages are:

1. avoids transform inversion gymnastics for every hit
   by using the inverse transform from the stree.iinst(double)
   instead of using the CSGFoundry qat4(float) transforms

2. double precision transform handling means the local positions
   suffer from less precision loss

3. avoids unidentified memory leak in the sframe.h/qat4.h
   transform handling

4. avoids complications from the sensor_identifier offsetting


Note that GPU side geometry has -1 fiddle : but that doesnt effect stree geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU side geometry must offset the sensor_identifier by one due to
optixInstance identifier range limitation. The offset being
done in the below geometry preparation methods::

   CSGFoundry::addInstance firstcall:true
   CSGFoundry::addInstanceVector


With Opticks-as-a-service doing localization server side would double hits transfer size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This suggests should do the localization in the client ? But that means the client
needs to have the Opticks geometry (specifically stree) to access the transforms.
Actually a robust client needs to form a geometry digest to ensure that its
using the same geometry as the server anyhow.

**/

void SEvt::getLocalHit(sphit& ht, sphoton& lp, unsigned idx) const
{
    getHit(lp, idx);   // copy *idx* hit from NP array (starts global frame) into sphoton& lp struct of caller
    int iindex = lp.iindex ;

    const glm::tmat4x4<double>* tr = tree ? tree->get_iinst(iindex) : nullptr ;

    LOG_IF(fatal, tr == nullptr)
         << " FAILED TO GET INSTANCE TRANSFORM : WHEN TESTING NEEDS SSim::Load NOT SSim::Create"
         << " iindex " << iindex
         << " tree " << ( tree ? "YES" : "NO " )
         << " tree.desc_inst " << ( tree ? tree->desc_inst() : "-" )
         ;
    assert( tr );

    bool normalize = true ;
    lp.transform( *tr, normalize );   // inplace transforms lp (pos, mom, pol) into local frame

    glm::tvec4<int64_t> col3 = {} ;
    strid::Decode( *tr, col3 );

    ht.iindex = col3[0] ;
    ht.sensor_identifier = col3[2] ;  // NB : NO "-1" HERE : SEE ABOVE COMMENT
    ht.sensor_index = col3[3] ;
}




/**
SEvt::getPhotonFrame  (TODO: replace with "getInstanceFrame" with iindex integer argument)
----------------------------------------------------------------------------------------------

Note that this relies on the photon iindex which
may not be set for photons ending in some places.
It should always be set for photons ending on PMTs
assuming properly instanced geometry.

The use of geometry information from this low level
struct is accomplished using the SGeo(cf) instance
that is fullfilled from higher level CSGFoundry
instance

**/

void SEvt::getPhotonFrame( sframe& fr, const sphoton& p ) const
{
    assert(cf);
    cf->getFrame(fr, p.iindex);
    fr.prepare();
}

std::string SEvt::descNum() const
{
    std::stringstream ss ;
    ss << "SEvt::descNum"
       << " getNumGenstepFromGenstep "  <<  getNumGenstepFromGenstep()
       << " getNumPhotonFromGenstep "   <<  getNumPhotonFromGenstep()
       << " getNumPhotonCollected "     <<  getNumPhotonCollected()
       << " getNumPhoton(from NPFold) " <<  getNumPhoton()
       << " getNumHit(from NPFold) "    <<  getNumHit()
       << std::endl
       ;

    std::string s = ss.str();
    return s ;
}

std::string SEvt::descPhoton(unsigned max_print) const
{
    unsigned num_photon = getNumPhoton();
    bool num_photon_unset =  (int)num_photon == -1 ;

    unsigned num_print = num_photon_unset ? 0 : std::min(max_print, num_photon);

    std::stringstream ss ;
    ss << "SEvt::descPhoton"
       << " num_photon " <<  num_photon
       << " num_photon_unset " <<  ( num_photon_unset ? "YES" : "NO " )
       << " max_print " <<  max_print
       << " num_print " <<  num_print
       << std::endl
       ;

    sphoton p ;
    for(unsigned idx=0 ; idx < num_print ; idx++)
    {
        getPhoton(p, idx);
        ss << p.desc() << std::endl  ;
    }

    std::string s = ss.str();
    return s ;
}

/**
SEvt::descLocalPhoton SEvt::descFramePhoton
----------------------------------------------

Note that SEvt::descLocalPhoton uses the frame of the
instance index of each photon whereas the SEvt::descFramePhoton
uses a single frame that is provided by SEvt::setFrame methods.

Hence caution wrt which frame is applicable for local photon.

**/

std::string SEvt::descLocalPhoton(unsigned max_print) const
{
    unsigned num_photon = getNumPhoton();
    bool num_photon_unset =  (int)num_photon == -1 ;
    unsigned num_print = num_photon_unset ? 0 : std::min(max_print, num_photon);

    std::stringstream ss ;
    ss << "SEvt::descLocalPhoton"
       << " num_photon " <<  num_photon
       << " num_photon_unset " <<  ( num_photon_unset ? "YES" : "NO " )
       << " max_print " <<  max_print
       << " num_print " <<  num_print
       << std::endl
       ;

    sphoton lp ;
    for(unsigned idx=0 ; idx < num_print ; idx++)
    {
        getLocalPhoton(lp, idx);
        ss << lp.desc() << std::endl  ;
    }
    std::string s = ss.str();
    return s ;
}

std::string SEvt::descFramePhoton(unsigned max_print) const
{
    unsigned num_photon = getNumPhoton();
    unsigned num_print = std::min(max_print, num_photon) ;
    bool zero_frame = frame.is_zero() ;

    std::stringstream ss ;
    ss << "SEvt::descFramePhoton"
       << " num_photon " <<  num_photon
       << " max_print " <<  max_print
       << " num_print " <<  num_print
       << " zero_frame " << zero_frame
       << std::endl
       ;

    if(zero_frame)
    {
        ss << "CANNOT getFramePhoton WITHOUT FRAME SET " << std::endl ;
    }
    else
    {
        sphoton fp ;
        for(unsigned idx=0 ; idx < num_print ; idx++)
        {
            getFramePhoton(fp, idx);
            ss << fp.desc() << std::endl  ;
        }
    }
    std::string s = ss.str();
    return s ;
}


std::string SEvt::descInputGenstep() const
{
    const char* ig = SEventConfig::InputGenstep() ;
    int c1 = 35 ;
    const char* div = " : " ;
    std::stringstream ss ;
    ss << "SEvt::descInputGenstep" << std::endl ;
    ss << std::setw(c1) << " SEventConfig::IntegrationMode "  << div << SEventConfig::IntegrationMode() << std::endl ;
    ss << std::setw(c1) << " SEventConfig::InputGenstep "      << div << ( ig  ? ig  : "-" ) << std::endl ;
    ss << std::setw(c1) << " input_genstep "   << div << ( input_genstep ? input_genstep->sstr() : "-" )     << std::endl ;
    ss << std::setw(c1) << " input_genstep.lpath " << div << ( input_genstep ? input_genstep->get_lpath() : "--" ) << std::endl ;
    std::string s = ss.str();
    return s ;
}

std::string SEvt::descInputPhoton() const
{
    const char* ip = SEventConfig::InputPhoton() ;
    const char* ipf = SEventConfig::InputPhotonFrame() ;
    int c1 = 35 ;

    const char* div = " : " ;
    std::stringstream ss ;
    ss << "SEvt::descInputPhoton" << std::endl ;
    ss << std::setw(c1) << " SEventConfig::IntegrationMode "  << div << SEventConfig::IntegrationMode() << std::endl ;
    ss << std::setw(c1) << " SEventConfig::InputPhoton "      << div << ( ip  ? ip  : "-" ) << std::endl ;
    ss << std::setw(c1) << " SEventConfig::InputPhotonFrame " << div << ( ipf ? ipf : "-" ) << std::endl ;
    ss << std::setw(c1) << " hasInputPhoton " << div << ( hasInputPhoton() ? "YES" : "NO " ) << std::endl ;
    ss << std::setw(c1) << " input_photon "   << div << ( input_photon ? input_photon->sstr() : "-" )     << std::endl ;
    ss << std::setw(c1) << " input_photon.lpath " << div << ( input_photon ? input_photon->get_lpath() : "--" ) << std::endl ;
    ss << std::setw(c1) << " hasInputPhotonTransformed " << div << ( hasInputPhotonTransformed() ? "YES" : "NO " ) ;
    std::string s = ss.str();
    return s ;
}

std::string SEvt::DescInputGenstep(int idx) // static
{
    return Exists(idx) ? Get(idx)->descInputGenstep() : "-" ;
}
std::string SEvt::DescInputPhoton(int idx) // static
{
    return Exists(idx) ? Get(idx)->descInputPhoton() : "-" ;
}






std::string SEvt::descFull(unsigned max_print) const
{
    std::stringstream ss ;
    ss << "[ SEvt::descFull "  << std::endl ;
    ss << ( cf ? cf->descBase() : "no-cf" ) << std::endl ;
    ss << descDir() << std::endl ;
    ss << descNum() << std::endl ;
    ss << descComponent() << std::endl ;
    ss << descInputPhoton() << std::endl ;

    ss << descPhoton(max_print) << std::endl ;
    ss << descLocalPhoton(max_print) << std::endl ;
    ss << descFramePhoton(max_print) << std::endl ;

    ss << ( cf ? cf->descBase() : "no-cf" ) << std::endl ;
    ss << ( topfold ? topfold->desc() : "no-topfold" ) << std::endl ;
    ss << "] SEvt::descFull "  << std::endl ;
    std::string s = ss.str();
    return s ;
}


/**
SEvt::getFramePhoton SEvt::getFrameHit
---------------------------------------

frame set by SEvt::setFrame is used to transform the photon into local "model" frame

**/

void SEvt::getFramePhoton(sphoton& lp, unsigned idx) const
{
    getPhoton(lp, idx);
    applyFrameTransform(lp);
}
void SEvt::getFrameHit(sphoton& lp, unsigned idx) const
{
    getHit(lp, idx);
    applyFrameTransform(lp);
}
void SEvt::applyFrameTransform(sphoton& lp) const
{
    bool zero_frame = frame.is_zero() ;
    LOG_IF(fatal, zero_frame) << " must setFrame before can applyFrameTransform " ;
    assert(!zero_frame);
    frame.transform_w2m(lp);
}

/**
SEvt::CountNibbles
--------------------

Create array of ints the same length as seq with nibble counts,
from 0 to 32(typically).

**/

NP* SEvt::CountNibbles( const NP* seq ) // static
{
    std::vector<sseq> qq ;
    NPX::VecFromArray<sseq>(qq, seq );
    int num_seq = qq.size();

    NP* seqnib = NP::Make<int>( seq->shape[0] ) ;
    int* nn = seqnib->values<int>() ;
    for(int i=0 ; i < num_seq ; i++)
    {
        const sseq& q = qq[i] ;
        nn[i] = q.seqhis_nibbles();
    }
    return seqnib ;
}

/**
SEvt::CountNibbles_Table
--------------------------

**/

NP* SEvt::CountNibbles_Table( const NP* seqnib ) // static
{
    int num_seqnib = seqnib->shape[0] ;
    const int* nn = seqnib->cvalues<int>() ;

    int ni =  sseq::SLOTS + 1 ;
    NP* seqnib_table = NP::Make<int>(ni, 1) ;
    int* cc = seqnib_table->values<int>() ;
    for(int i=0 ; i < num_seqnib ; i++)
    {
        int nibs = nn[i] ;
        assert( nibs < ni );
        cc[nibs] += 1 ;
    }
    return seqnib_table ;
}






