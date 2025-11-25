
#include <csignal>

#include "SLOG.hh"

#include "ssys.h"
#include "sstamp.h"
#include "spath.h"
#include "SProf.hh"

#include "SComp.h"
#include "SEvt.hh"
#include "SSim.hh"
#include "scuda.h"
#include "squad.h"
#include "salloc.h"
#include "SEvent.hh"
#include "SEventConfig.hh"
#include "SCSGOptiX.h"

#include "SGenstep.h"
#include "sslice.h"

#include "NP.hh"
#include "QUDA_CHECK.h"
#include "QU.hh"

#include "qrng.h"
#include "qsim.h"
#include "qdebug.h"

#include "QBase.hh"
#include "QEvt.hh"
#include "QRng.hh"
#include "QTex.hh"
#include "QScint.hh"
#include "QCerenkov.hh"
#include "QBnd.hh"
#include "QProp.hh"
#include "QMultiFilm.hh"
#include "QEvt.hh"
#include "QOptical.hh"
#include "QSimLaunch.hh"
#include "QDebug.hh"
#include "QPMT.hh"

#include "QSim.hh"

const plog::Severity QSim::LEVEL = SLOG::EnvLevel("QSim", "DEBUG");

const bool  QSim::REQUIRE_PMT = ssys::getenvbool(_QSim__REQUIRE_PMT);
const int   QSim::SAVE_IGS_EVENTID = ssys::getenvint(_QSim__SAVE_IGS_EVENTID,-1) ;
const char* QSim::SAVE_IGS_PATH = ssys::getenvvar(_QSim__SAVE_IGS_PATH, "$TMP/.opticks/igs.npy");
const bool  QSim::CONCAT = ssys::getenvbool(_QSim__CONCAT);
const bool  QSim::ALLOC  = ssys::getenvbool(_QSim__ALLOC);



QSim* QSim::INSTANCE = nullptr ;
QSim* QSim::Get(){ return INSTANCE ; }

QSim* QSim::Create()
{
    LOG_IF(fatal, INSTANCE != nullptr) << " a QSim INSTANCE already exists " ;
    assert( INSTANCE == nullptr ) ;
    return new QSim  ;
}




/**
QSim::UploadComponents
-----------------------

Canonical invokation from CSGOptiX::InitSim by CSGOptiX::Create

As the QBase instanciation done by QSim::UploadComponents
is typically the first connection to the GPU device
by an Opticks process it is prone to CUDA latency of
order 1.5s per GPU if nvidia-persistenced is not running.
Start the daemon to avoid this latency following:

* https://docs.nvidia.com/deploy/driver-persistence/index.html

Essentially::

    N[blyth@localhost ~]$ sudo su
    N[root@localhost blyth]$ mkdir -p /var/run/nvidia-persistenced
    N[root@localhost blyth]$ chown blyth:blyth /var/run/nvidia-persistenced
    N[root@localhost blyth]$ which nvidia-persistenced
    /bin/nvidia-persistenced
    N[root@localhost blyth]$ nvidia-persistenced --user blyth
    N[root@localhost blyth]$ ls /var/run/nvidia-persistenced/
    nvidia-persistenced.pid  socket


QSim::UploadComponents is invoked for example by CSGOptiX/tests/CSGOptiXSimtraceTest.cc
prior to instanciating CSGOptiX

Uploading components is a once only action for a geometry, encompassing:

* random states
* scintillation textures
* boundary textures
* property arrays

It is the simulation physics equivalent of uploading the CSGFoundry geometry.

The components are managed by separate singleton instances
that subsequent QSim instanciation collects together.
This structure is used to allow separate testing.

**/

void QSim::UploadComponents( const SSim* ssim  )
{
    LOG(LEVEL) << "[ ssim " << ssim ;
    if(getenv("QSim__UploadComponents_SIGINT")) std::raise(SIGINT);

    LOG(LEVEL) << "[ new QBase" ;
    QBase* base = new QBase ;
    LOG(LEVEL) << "] new QBase : latency here of about 0.3s from first device access, if latency of >1s need to start nvidia-persistenced " ;
    LOG(LEVEL) << base->desc();


    unsigned skipahead_event_offset = SEventConfig::EventSkipahead()  ;
    LOG(LEVEL) << "[ new QRng skipahead_event_offset : " << skipahead_event_offset << " " << SEventConfig::kEventSkipahead ;
    QRng* rng = new QRng(skipahead_event_offset)  ;  // loads and uploads RNG
    LOG(LEVEL) << "] new QRng " << rng->desc()  ;

    LOG(LEVEL) << rng->desc();

    const NP* optical = ssim->get(snam::OPTICAL);
    const NP* bnd = ssim->get(snam::BND);

    if( optical == nullptr && bnd == nullptr )
    {
        LOG(error) << " optical and bnd null  snam::OPTICAL " << snam::OPTICAL << " snam::BND " << snam::BND  ;
    }
    else
    {
       // note that QOptical and QBnd are tightly coupled, perhaps add constraints to tie them together
        QOptical* qopt = new QOptical(optical);
        LOG(LEVEL) << qopt->desc();

        QBnd* qbnd = new QBnd(bnd); // boundary texture with standard domain, used for standard fast property lookup
        LOG(LEVEL) << qbnd->desc();
    }

    QDebug* debug_ = new QDebug ;
    LOG(LEVEL) << debug_->desc() ;

    const NP* propcom = ssim->get(snam::PROPCOM);
    if( propcom )
    {
        LOG(LEVEL) << "[ QProp " ;
        QProp<float>* prop = new QProp<float>(propcom) ;
        // property interpolation with per-property domains, eg used for Cerenkov RINDEX sampling
        LOG(LEVEL) << "] QProp " ;
        LOG(LEVEL) << prop->desc();
    }
    else
    {
        LOG(LEVEL) << "  propcom null, snam::PROPCOM " <<  snam::PROPCOM ;
    }


    const NP* icdf = ssim->get(snam::ICDF);
    if( icdf == nullptr )
    {
        LOG(error) << " icdf null, snam::ICDF " << snam::ICDF ;
    }
    else
    {
        unsigned hd_factor = 20u ;  // 0,10,20
        QScint* scint = new QScint( icdf, hd_factor); // custom high-definition inverse CDF for scintillation generation
        LOG(LEVEL) << scint->desc();
    }


    // TODO: make this more like the others : acting on the available inputs rather than the mode
    bool is_simtrace = SEventConfig::IsRGModeSimtrace() ;
    if(is_simtrace == false )
    {
        QCerenkov* cerenkov = new QCerenkov  ;
        LOG(LEVEL) << cerenkov->desc();
    }
    else
    {
        LOG(LEVEL) << " skip QCerenkov for simtrace running " ;
    }




    const NPFold* spmt_f = ssim->get_spmt_f() ;
    QPMT<float>* qpmt = spmt_f ? new QPMT<float>(spmt_f) : nullptr ;

    bool has_PMT = spmt_f != nullptr && qpmt != nullptr ;
    bool MISSING_PMT = REQUIRE_PMT == true && has_PMT == false ;

    LOG_IF(fatal, MISSING_PMT )
        << " MISSING_PMT "
        << " has_PMT " << ( has_PMT ? "YES" : "NO " )
        << " REQUIRE_PMT " << ( REQUIRE_PMT ? "YES" : "NO " )
        << " MISSING_PMT " << ( MISSING_PMT ? "YES" : "NO " )
        << " spmt_f " << ( spmt_f ? "YES" : "NO " )
        << " qpmt " << ( qpmt ? "YES" : "NO " )
        ;

    assert(MISSING_PMT == false) ;
    if(MISSING_PMT)  std::raise(SIGINT);



    LOG(LEVEL)
        << QPMT<float>::Desc()
        << std::endl
        << " spmt_f " << ( spmt_f ? "YES" : "NO " )
        << " qpmt " << ( qpmt ? "YES" : "NO " )
        ;



    const NP* multifilm = ssim->get_extra(snam::MULTIFILM);
    if(multifilm == nullptr)
    {
        LOG(LEVEL) << " multifilm null, snam::MULTIFILM " << snam::MULTIFILM ;
    }
    else
    {
        QMultiFilm* mul = new QMultiFilm( multifilm );
        LOG(LEVEL) << mul->desc();
    }
    LOG(LEVEL) << "] ssim " << ssim ;



}






/**
QSim:::QSim
-------------

Canonical instance is instanciated with CSGOptiX::CSGOptiX in sim mode.
Notice how the instanciation pulls together device pointers from
the constituents into the CPU side *sim* and then uploads that to *d_sim*
which is available as *sim* GPU side.

Prior to instanciating QSim invoke QSim::Init to prepare the
singleton components.

**/

QSim::QSim()
    :
    base(QBase::Get()),
    qev(new QEvt),
    sev(qev->sev),
    rng(QRng::Get()),
    scint(QScint::Get()),
    cerenkov(QCerenkov::Get()),
    bnd(QBnd::Get()),
    debug_(QDebug::Get()),
    prop(QProp<float>::Get()),
    pmt(QPMT<float>::Get()),
    multifilm(QMultiFilm::Get()),
    sim(nullptr),
    d_sim(nullptr),
    dbg(debug_ ? debug_->dbg : nullptr),
    d_dbg(debug_ ? debug_->d_dbg : nullptr),
    cx(nullptr)
{
    LOG(LEVEL) << desc() ;
    init();
}


/**
QSim::init
------------

*sim* (qsim.h) is a host side instance that is populated
with device side pointers and handles and then uploaded
to the device *d_sim*.

Many device pointers and handles are then accessible from
the qsim.h instance at the cost of only a single launch
parameter argument: the d_sim pointer
or with optix launches a single Params member.

The advantage of this approach is it avoids kernel
launches having very long argument lists and provides a natural
place (qsim.h) to add GPU side functionality.

**/


void QSim::init()
{
    sim = new qsim ;
    sim->base = base ? base->d_base : nullptr ;
    sim->evt = qev ? qev->getDevicePtr() : nullptr ;
    //sim->rng_state = rng ? rng->qr->uploaded_states : nullptr ;
    sim->rng = rng ? rng->d_qr : nullptr ;

    sim->bnd = bnd ? bnd->d_qb : nullptr ;
    sim->multifilm = multifilm ? multifilm->d_multifilm : nullptr ;
    sim->cerenkov = cerenkov ? cerenkov->d_cerenkov : nullptr ;
    sim->scint = scint ? scint->d_scint : nullptr ;
    sim->pmt = pmt ? pmt->d_pmt : nullptr ;


    bool has_PMT = pmt != nullptr && sim->pmt != nullptr ;
    bool REQUIRE_PMT = ssys::getenvbool(_QSim__REQUIRE_PMT);
    bool MISSING_PMT = REQUIRE_PMT == true && has_PMT == false ;

    LOG(LEVEL)
        << " MISSING_PMT " << ( MISSING_PMT ? "YES" : "NO " )
        << " has_PMT " << ( has_PMT ? "YES" : "NO " )
        << " QSim::pmt " << ( pmt ? "YES" : "NO " )
        << " QSim::pmt->d_pmt " << ( sim->pmt ? "YES" : "NO " )
        << " [" << _QSim__REQUIRE_PMT << "] " << ( REQUIRE_PMT ? "YES" : "NO " )
        ;

    LOG_IF(fatal, MISSING_PMT )
        << " MISSING_PMT ABORT "
        << " MISSING_PMT " << ( MISSING_PMT ? "YES" : "NO " )
        << " has_PMT " << ( has_PMT ? "YES" : "NO " )
        << " QSim::pmt " << ( pmt ? "YES" : "NO " )
        << " QSim::pmt->d_pmt " << ( sim->pmt ? "YES" : "NO " )
        << " [" << _QSim__REQUIRE_PMT << "] " << ( REQUIRE_PMT ? "YES" : "NO " )
        ;

    assert(MISSING_PMT == false) ;
    if(MISSING_PMT)  std::raise(SIGINT);

    d_sim = QU::UploadArray<qsim>(sim, 1, "QSim::init.sim" );

    INSTANCE = this ;
    LOG(LEVEL) << desc() ;
    LOG(LEVEL) << descComponents() ;
}

/**
QSim::setLauncher

**/
void QSim::setLauncher(SCSGOptiX* cx_ )
{
    cx = cx_ ;
}


/**
QSim::post_launch
--------------------

launch is async, so need to sync before can gather ?
HOW ELSE TO DO THIS : LIKELY BIG PERFORMANCE EFFECT

cudaDeviceSynchronize already done by CSGOptiX::launch via CUDA_SYNC_CHECK see CSG/CUDA_CHECK.h

void QSim::post_launch()
{
    cudaDeviceSynchronize();
}
**/


/**
QSim::simulate
---------------

This expects gensteps to have been collected into SEvt vectors of quad6
prior to being called.

Now only one canonical invokations, the higher level G4CXOpticks no longer
directly uses QSim:

1. CSGOptiX::simulate for pure CSGOptiX level testing such as by ~/o/cxs_min.sh


Collected genstep are uploaded and the CSGOptiX kernel is launched to generate and propagate.

NB the surprising fact that this calls CSGOptiX::simulate_launch (using a protocol),
that seems funny dependency-wise but its needed for genstep preparation prior to
the launch.

::

    QSim::simulate

       EGPU.SEvt::beginOfEvent

       determine slices over gensteps depending on VRAM

       for each slice

           QEvt::setGenstepUpload_NP

           SCSGOptiX::simulate_launch

           SEvt::gather into *fold* below *topfold*

       end for each slice

       (NPFold)topfold->concat joining all arrays from the per-launch *fold* into the one *topfold*

       EGPU.SEvt::endOfEvent [only when reset:true, not so in OJ running as need to defer until after hits collected]


**/

bool QSim::KEEP_SUBFOLD = ssys::getenvbool(QSim__simulate_KEEP_SUBFOLD);

double QSim::simulate(int eventID, bool reset_)
{
    SProf::SetTag(eventID, "A%0.3d_" ) ;

    assert( SEventConfig::IsRGModeSimulate() );

    //cudaStream_t stream ;  cudaStreamCreate(&stream);
    cudaStream_t stream = 0 ;


    int64_t tot_ph = 0 ;

    double tot_dt = 0. ;

    int64_t tot_idt = 0 ;
    int64_t tot_gdt = 0 ;

    int64_t t_HEAD = SProf::Add("QSim__simulate_HEAD");

    LOG_IF(info, SEvt::LIFECYCLE) << "[ eventID " << eventID ;
    if( qev == nullptr ) return -1. ;


    sev->beginOfEvent(eventID);  // set SEvt index and tees up frame gensteps for simtrace and input photon simulate running

    NP* igs = sev->makeGenstepArrayFromVector();

    MaybeSaveIGS(eventID, igs);

    std::vector<sslice> igs_slice ;
    int64_t tot_ph_0 = SGenstep::GetGenstepSlices( igs_slice, igs, SEventConfig::MaxSlot() );

    //bool xxl = tot_ph_0 > SGenstep::MAX_SLOT_PER_SLICE ;
    bool xxl = tot_ph_0 > 100*M ;

    int num_slice = igs_slice.size();

    LOG(xxl ? info : LEVEL)
        << " eventID " << std::setw(6) << eventID
        << " igs " << ( igs ? igs->sstr() : "-" )
        << " tot_ph_0 " << tot_ph_0
        << " tot_ph_0/M " << tot_ph_0/M
        << " xxl " << ( xxl ? "YES" : "NO " )
        << " MaxSlot " << SEventConfig::MaxSlot()
        << " MaxSlot/M " << SEventConfig::MaxSlot()/M
        << " sslice::Desc(igs_slice)\n"
        << sslice::Desc(igs_slice)
        << " num_slice " << num_slice
        ;


    int64_t t_LBEG = SProf::Add("QSim__simulate_LBEG");

    for(int i=0 ; i < num_slice ; i++)
    {
        SProf::Add("QSim__simulate_PRUP");

        const sslice& sl = igs_slice[i] ;

        LOG(LEVEL) << sl.idx_desc(i) ;

        int rc = qev->setGenstepUpload_NP(igs, &sl ) ;
        LOG_IF(error, rc != 0) << " QEvt::setGenstep ERROR : have qev but no gensteps collected : will skip cx.simulate " ;

        LOG_IF(info, ALLOC)
            << " [" << _QSim__ALLOC << "] "
            << " i " << std::setw(5) << i
            << " SEventConfig::ALLOC " << ( SEventConfig::ALLOC  ? "YES" : "NO " )
            << ( SEventConfig::ALLOC ? SEventConfig::ALLOC->desc() : "-" )
            ;


        SProf::Add("QSim__simulate_PREL");

        sev->t_PreLaunch = sstamp::Now() ;

        double dt = rc == 0 && cx != nullptr ? cx->simulate_launch() : -1. ;  //SCSGOptiX protocol

        sev->t_PostLaunch = sstamp::Now() ;
        sev->t_Launch = dt ;

        tot_idt += ( sev->t_PostLaunch - sev->t_PreLaunch ) ;
        tot_dt += dt ;
        tot_ph += sl.ph_count ;

        LOG( xxl ? info : LEVEL )
            << " eventID " << eventID
            << " xxl " << ( xxl ? "YES" : "NO " )
            << " i " << std::setw(4) << i
            << " dt " << std::setw(11) << std::fixed << std::setprecision(6) << dt
            << " slice " << sl.idx_desc(i)
            ;

        int64_t t_POST = SProf::Add("QSim__simulate_POST");

        sev->gather();  // gather into *fold* just added to *topfold*

        int64_t t_DOWN = SProf::Add("QSim__simulate_DOWN");

        tot_gdt += ( t_DOWN - t_POST ) ;
    }

    int64_t t_LEND = SProf::Add("QSim__simulate_LEND");

    std::stringstream ss ;
    std::ostream* out = CONCAT ? &ss : nullptr ;
    int concat_rc = sev->topfold->concat(out);

    LOG_IF(info, CONCAT) << ss.str() ;
    LOG_IF(fatal, concat_rc != 0) << " sev->topfold->concat FAILED " ;
    assert(concat_rc == 0);

    bool has_hlm = sev->topfold->has_key(SComp::HITLITEMERGED_);
    bool hlm_final_merge = num_slice > 1 && has_hlm ;
    LOG(LEVEL)
         << " num_slice " << num_slice
         << " has_hlm " << ( has_hlm ? "YES" : "NO " )
         << " hlm_final_merge " << ( hlm_final_merge ? "YES" : "NO " )
         ;
    if(hlm_final_merge) simulate_final_merge(tot_ph, stream);


    if(!KEEP_SUBFOLD) sev->topfold->clear_subfold();

    int64_t t_PCAT = SProf::Add("QSim__simulate_PCAT");

    int tot_ht = sev->getNumHit() ;  // NB from fold, so requires hits array gathering to be configured to get non-zero
    std::string counts = sev->getCounts();  // collect counts before reset

    LOG_IF(info, SEvt::MINIMAL)
        << " eventID " << eventID
        << " tot_dt " << std::setw(11) << std::fixed << std::setprecision(6) << tot_dt
        << " tot_ph " << std::setw(10) << tot_ph
        << " tot_ph/M " << std::setw(10) << std::fixed << std::setprecision(6) << float(tot_ph)/float(M)
        << " tot_ht " << std::setw(10) << tot_ht
        << " tot_ht/M " << std::setw(10) << std::fixed << std::setprecision(6) << float(tot_ht)/float(M)
        << " tot_ht/tot_ph " << std::setw(10) << std::fixed << std::setprecision(6) << float(tot_ht)/float(tot_ph)
        << " reset_ " << ( reset_ ? "YES" : "NO " )
        ;


    assert( tot_ph == tot_ph_0 );

    int64_t t_BRES  = SProf::Add("QSim__simulate_BRES", counts.c_str() );
    if(reset_) reset(eventID) ;

    int64_t t_TAIL  = SProf::Add("QSim__simulate_TAIL");

    SProf::Write(); // per-event write, so have something in case of crash

    LOG_IF(info, SEvt::MINTIME) << "\n"
        << SEvt::SEvt__MINTIME
        << "\n"
        << " (TAIL - HEAD)/M " << std::setw(10) << std::fixed << std::setprecision(6) << float( t_TAIL - t_HEAD )/M
        << " (head to tail of QSim::simulate method) "
        << "\n"
        << " (LEND - LBEG)/M " << std::setw(10) << std::fixed << std::setprecision(6) << float( t_LEND - t_LBEG )/M
        << " (multilaunch loop begin to end) "
        << "\n"
        << " (PCAT - LEND)/M " << std::setw(10) << std::fixed << std::setprecision(6) << float( t_PCAT - t_LEND )/M
        << " (topfold concat and clear subfold) "
        << "\n"
        << " (TAIL - BRES)/M " << std::setw(10) << std::fixed << std::setprecision(6) << float( t_TAIL - t_BRES )/M
        << " (QSim::reset which saves hits) "
        << "\n"
        << " tot_idt/M       " << std::setw(10) << std::fixed << std::setprecision(6) << float(tot_idt)/M
        << " (sum of kernel execution int64_t stamp differences in microseconds)"
        << "\n"
        << " tot_dt          " << std::setw(10) << std::fixed << std::setprecision(6) << tot_dt
        << " int(tot_dt*M)   " << std::setw(10) << int64_t(tot_dt*M)
        << " (sum of kernel execution double chrono stamp differences in seconds, and scaled to ms) "
        << "\n"
        << " tot_gdt/M       " << std::setw(10) << std::fixed << std::setprecision(6) << float(tot_gdt)/M
        << " (sum of SEvt::gather int64_t stamp differences in microseconds)"
        << "\n"
        ;

    return tot_dt ;
}


/**
QSim::simulate_final_merge
---------------------------

* NB this normally only runs for large events with more
  photons than fit in VRAM that cause multiple launches
  to be done

* requires topfold to have "hitlitemerged" array,
  expected to be the result of simple concatenation
  of individual launch "hitlitemerged" arrays

* does CUDA thrust implemented hitlitemerged final merging


TODO: use QEvt::FinalMerge_async once that makes sense

**/

void QSim::simulate_final_merge(int64_t tot_ph, cudaStream_t stream)
{
    bool has_hlm = sev->topfold->has_key(SComp::HITLITEMERGED_);
    assert( has_hlm );

    const NP* hlm = sev->topfold->get(SComp::HITLITEMERGED_);
    NP*       fin = QEvt::FinalMerge(hlm, stream);

    float     hlm_frac = float(hlm->num_items())/float(tot_ph) ;
    float     fin_frac = float(fin->num_items())/float(hlm->num_items()) ;

    std::stringstream ss ;
    ss
        << " tot_ph " << tot_ph
        << " hlm " << ( hlm ? hlm->sstr() : "-" )
        << " fin " << ( fin ? fin->sstr() : "-" )
        << " hlm/tot " << std::setw(7) << std::fixed << std::setprecision(4) << hlm_frac
        << " fin/hlm " << std::setw(7) << std::fixed << std::setprecision(4) << fin_frac
        ;

    std::string note = ss.str();
    fin->set_meta<std::string>("QSim__simulate_final_merge", note );

    sev->topfold->set(SComp::HITLITEMERGED_, fin );

    LOG(info) << note ;
}


/**
QSim::simulate
----------------

High level API intending to be used from CSGOptiXService

1. access eventID from genstep metadata
2. add the genstep to the EGPU SEvt
3. invoke simulate doing the kernel launch and pullback
4. copy global hits array (current thinking that localization
   should happen within the client to avoid ~doubling hits transfer size)
5. reset the SEvt doing dealloc

Thus is used from language crossing stack::

    async def simulate(request: Request)  [CSGOptiX/tests/CSGOptiXService_FastAPI_test/main.py]
    inline nb::ndarray<nb::numpy> _CSGOptiXService::simulate( nb::ndarray<nb::numpy> _gs, int eventID )
    inline NP* CSGOptiXService::simulate( NP* gs, int eventID )
    NP* CSGOptiX::simulate(const NP* gs, int eventID)

**/


NP* QSim::simulate(const NP* gs, int eventID )
{
    bool eventID_expected = eventID > -1;
    if(!eventID_expected) std::cerr << "QSim::simulate gs lacks needed eventID metadata [" << eventID << "]\n" ;
    assert(eventID_expected);

    assert( sev == SEvt::Get_EGPU() );
    sev->addGenstep(gs);

    bool reset_ = false ;
    double tot_dt = simulate(eventID, reset_);

    const NP* _ht = sev->getHit();
    NP* ht = _ht ? _ht->copy() : nullptr ;  // copy global hits from SEvt before reset
    ht->set_meta<double>("QSim__simulate_tot_dt", tot_dt);

    LOG(info)
        << " eventID " << std::setw(6) << eventID
        << " gs " << ( gs ? gs->sstr() : "-" )
        << " ht " << ( ht ? ht->sstr() : "-" )
        << " tot_dt " << std::fixed << std::setw(10) << std::setprecision(6) << tot_dt
        ;
    reset(eventID);

    return ht ;
}



/**
QSim::MaybeSaveIGS
--------------------

Invoked from QSim::simulate just prior to the launch loop
over genstep slices. To configure saving of the gensteps
use the below envvars::

    export QSim__SAVE_IGS_EVENTID=6                   ## default -1 does not save
    export QSim__SAVE_IGS_PATH=$TMP/.opticks/igs.npy  ## default path below $TMP

Saving gensteps prior to launch is useful for eventID that cause crashes,
as the saved gensteps can then be used with input genstep running::

    export OPTICKS_RUNNING_MODE=SRM_INPUT_GENSTEP

That enables fast cycle rerunning without Geant4 initalization, see eg::

    TEST=input_genstep_muon cxs_min.sh

A potential cause of crashes is the heuristic OPTICKS_MAX_SLOT
determined based on available VRAM is too optimistically large.
Try manually reducing slots to see if memory limits are the cause::

    export OPTICKS_MAX_SLOT=M100

**/

void QSim::MaybeSaveIGS(int eventID, NP* igs) // static
{
    bool igs_null = igs == nullptr ;
    const char* igs_path = SAVE_IGS_PATH ? spath::Resolve(SAVE_IGS_PATH) : nullptr ;
    bool save_igs = igs && SAVE_IGS_EVENTID == eventID && igs_path ;
    LOG(LEVEL)
        << " eventID " << eventID
        << " igs " << ( igs ? igs->sstr() : "-" )
        << " igs_null " << ( igs_null ? "YES" : "NO " )
        << " [" << _QSim__SAVE_IGS_EVENTID << "] " <<  SAVE_IGS_EVENTID
        << " [" << _QSim__SAVE_IGS_PATH    << "] " << ( SAVE_IGS_PATH ? SAVE_IGS_PATH : "-" )
        << " igs_path [" << ( igs_path ? igs_path : "-" ) << "]"
        << " save_igs " << ( save_igs ? "YES" : "NO " )
        ;

    if(!save_igs) return ;
    igs->save(igs_path);
}



/**
QSim::getPhotonSlotOffset
---------------------------

The photon slot offset is available
from QEvt after setGenstepUpload_NP

The first genstep slice yields an offset of zero.
So this only yields non-zero offsets for multi-launch
running,


The photon_slot_offset is uploaded to Params on device prior
to simulate launches by CSGOptiX::prepareParamSimulate

The offset is used at head of CSGOptiX7.cu:simulate  to offset the
launch index.

Hence the sum of the offset and the
number of photons in the launch must be less than
or equal the number of states uploaded.

**/



unsigned long long QSim::get_photon_slot_offset() const
{
    return qev->get_photon_slot_offset() ;
}


/**
QSim::reset
------------

When *QSim::simulate* is called with argument *reset:true* the
*QSim::reset* method is called which invokes SEvt::endOfEvent in order
to clean up the SEvt. Normally that would be done after saving
any Opticks configured arrays.

When *QSim::simulate* is called with argument *reset:false*
the *QSim::reset* method must be called separately in order to avoid a memory leak.
Using *reset:false* is typically done in order to keep arrays alive longer
to enable copying from the gathered arrays into non-Opticks collections.

**/
void QSim::reset(int eventID)
{
    SProf::Add("QSim__reset_HEAD");
    qev->clear();
    sev->endOfEvent(eventID);
    LOG_IF(info, SEvt::LIFECYCLE) << "] eventID " << eventID ;
    SProf::Add("QSim__reset_TAIL");
}



/**
QSim::simtrace
---------------

Canonically invoked from G4CXOpticks::simtrace
Collected genstep are uploaded and the CSGOptiX kernel is launched to generate and propagate.

**/


double QSim::simtrace(int eventID)
{
    assert( SEventConfig::IsRGModeSimtrace() );


    sev->beginOfEvent(eventID);

    NP* igs = sev->makeGenstepArrayFromVector();

    LOG_IF(fatal, igs==nullptr)
         << " igs NULL "
         << " sev.descGenstepArrayFromVector " << sev->descGenstepArrayFromVector()
         ;

    assert(igs);
    int rc = qev->setGenstepUpload_NP(igs) ;

    LOG_IF(error, rc != 0) << " QEvt::setGenstep ERROR : no gensteps collected : will skip cx.simtrace " ;

    sev->t_PreLaunch = sstamp::Now() ;
    double dt = rc == 0 && cx != nullptr ? cx->simtrace_launch() : -1. ;
    sev->t_PostLaunch = sstamp::Now() ;
    sev->t_Launch = dt ;

    // see ~/o/notes/issues/cxt_min_simtrace_revival.rst
    sev->gather();

    sev->topfold->concat();
    sev->topfold->clear_subfold();

    sev->endOfEvent(eventID);

    return dt ;
}


qsim* QSim::getDevicePtr() const
{
    return d_sim ;
}


char QSim::getScintTexFilterMode() const
{
    return scint->tex->getFilterMode() ;
}

std::string QSim::desc() const
{
    std::stringstream ss ;
    ss << "QSim::desc"
       << std::endl
       << " this 0x"            << std::hex << std::uint64_t(this)     << std::dec
       << " INSTANCE 0x"        << std::hex << std::uint64_t(INSTANCE) << std::dec
       << " QEvt.hh:qev 0x" << std::hex << std::uint64_t(qev)    << std::dec
       << " qsim.h:sim 0x"      << std::hex << std::uint64_t(sim)      << std::dec
       ;
    std::string s = ss.str();
    return s ;
}

std::string QSim::descFull() const
{
    std::stringstream ss ;
    ss
       << std::endl
       << "QSim::descFull"
       << std::endl
       << " this 0x"            << std::hex << std::uint64_t(this)     << std::dec
       << " INSTANCE 0x"        << std::hex << std::uint64_t(INSTANCE) << std::dec
       << " QEvt.hh:qev 0x" << std::hex << std::uint64_t(qev)    << std::dec
       << " qsim.h:sim 0x"      << std::hex << std::uint64_t(sim)      << std::dec
       << " qsim.h:d_sim 0x"    << std::hex << std::uint64_t(d_sim)    << std::dec
       //<< " sim->rng_state 0x"   << std::hex << std::uint64_t(sim->rng_state) << std::dec  // tending to SEGV on some systems
       << " sim->base 0x"       << std::hex << std::uint64_t(sim->base)  << std::dec
       << " sim->bnd 0x"        << std::hex << std::uint64_t(sim->bnd)   << std::dec
       << " sim->scint 0x"      << std::hex << std::uint64_t(sim->scint) << std::dec
       << " sim->cerenkov 0x"   << std::hex << std::uint64_t(sim->cerenkov) << std::dec
       ;
    std::string s = ss.str();
    return s ;
}

std::string QSim::descComponents() const
{
    std::stringstream ss ;
    ss << std::endl
       << "QSim::descComponents"
       << std::endl
       << " (QBase)base             " << ( base      ? "YES" : "NO " )  << std::endl
       << " (QEvt)qev           " << ( qev     ? "YES" : "NO " )  << std::endl
       << " (SEvt)sev               " << ( sev       ? "YES" : "NO " )  << std::endl
       << " (QRng)rng               " << ( rng       ? "YES" : "NO " )  << std::endl
       << " (QScint)scint           " << ( scint     ? "YES" : "NO " )  << std::endl
       << " (QCerenkov)cerenkov     " << ( cerenkov  ? "YES" : "NO " )  << std::endl
       << " (QBnd)bnd               " << ( bnd       ? "YES" : "NO " )  << std::endl
       << " (QOptical)optical       " << ( optical   ? "YES" : "NO " )  << std::endl
       << " (QDebug)debug_          " << ( debug_    ? "YES" : "NO " )  << std::endl
       << " (QProp)prop             " << ( prop      ? "YES" : "NO " )  << std::endl
       << " (QPMT)pmt               " << ( pmt       ? "YES" : "NO " )  << std::endl
       << " (QMultiFilm)multifilm   " << ( multifilm ? "YES" : "NO " )  << std::endl
       << " (qsim)sim               " << ( sim       ? "YES" : "NO " )  << std::endl
       << " (qsim)d_sim             " << ( d_sim     ? "YES" : "NO " )  << std::endl
       << " (qdebug)dbg             " << ( dbg       ? "YES" : "NO " )  << std::endl
       << " (qdebug)d_dbg           " << ( d_dbg     ? "YES" : "NO " )  << std::endl
       ;
    std::string s = ss.str();
    return s ;
}





void QSim::configureLaunch(unsigned width, unsigned height )
{
    QU::ConfigureLaunch(numBlocks, threadsPerBlock, width, height);
}

void QSim::configureLaunch2D(unsigned width, unsigned height )
{
    QU::ConfigureLaunch2D(numBlocks, threadsPerBlock, width, height);
}

void QSim::configureLaunch16()
{
    QU::ConfigureLaunch16(numBlocks, threadsPerBlock);
}

void QSim::configureLaunch1D(unsigned num, unsigned threads_per_block)
{
    QU::ConfigureLaunch1D(numBlocks, threadsPerBlock, num, threads_per_block);
}


std::string QSim::descLaunch() const
{
    return QU::DescLaunch(numBlocks, threadsPerBlock);
}









/**
QSim::rng_sequence mass production with multiple launches...
--------------------------------------------------------------

The output files are split too::

    epsilon:opticks blyth$ np.py *.npy
    a :                                            TRngBufTest_0.npy :      (10000, 16, 16) : 8f9b27c9416a0121574730baa742b5c9 : 20210715-1227
    epsilon:opticks blyth$ du -h TRngBufTest_0.npy
     20M	TRngBufTest_0.npy

    In [6]: (16*16*4*2*10000)/1e6
    Out[6]: 20.48

Upping to 1M would be 100x 20M = 2000M  2GB

* using floats would half storage to 1GB, just promote to double in cks/OpticksRandom::flat
  THIS MAKES SENSE AS curand_uniform IS GENERATING float anyhow
* 100k blocks would mean 10*100Mb files for 1M : can do in 10 launches to work on any GPU

**/


template <typename T>
extern void QSim_rng_sequence(  dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, T* seq, unsigned ni, unsigned nj, unsigned id_offset );


/**
QSim::rng_sequence generate randoms in single CUDA launch
-------------------------------------------------------------

This is invoked for each tranche by the below rng_sequence method.
Each tranche launch generates ni_tranche*nv randoms writing them into seq

ni_tranche : item tranche size

nv : number randoms to generate for each item

id_offset : acts on the rng_state array

skipahead : used curand skipahead offsets depending on sim->evt->index and OPTICKS_EVENT_SKIPAHEAD


**/

template <typename T>
void QSim::rng_sequence( T* seq, unsigned ni_tranche, unsigned nv, unsigned id_offset )
{
    configureLaunch(ni_tranche, 1 );

    unsigned num_rng = ni_tranche*nv ;

    const char* label = "QSim::rng_sequence:num_rng" ;

    T* d_seq = QU::device_alloc<T>(num_rng, label );

    QSim_rng_sequence<T>( numBlocks, threadsPerBlock, d_sim, d_seq, ni_tranche, nv, id_offset );

    QU::copy_device_to_host_and_free<T>( seq, d_seq, num_rng, label );
}



const char* QSim::PREFIX = "rng_sequence" ;

/**
QSim::rng_sequence
---------------------

*ni* is the total number of items across all launches that is
split into tranches of *ni_tranche_size* each.

As separate CPU and GPU memory allocations, launches and output files are made
for each tranche the total generated size may exceed the total available memory of the GPU.
Splitting the output files also eases management by avoiding huge files.

Default *dir* is $TMP/QSimTest/rng_sequence leading to npy paths like::

    /tmp/blyth/opticks/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy

**/

template <typename T>
void QSim::rng_sequence( const char* dir, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size )
{
    assert( ni >= ni_tranche_size && ni % ni_tranche_size == 0 );   // total size *ni* must be integral multiple of *ni_tranche_size*
    unsigned num_tranche = ni/ni_tranche_size ;
    unsigned nv = nj*nk ;

    unsigned size = ni_tranche_size*nv ;   // number of randoms to be generated in each launch
    std::string reldir = QU::rng_sequence_reldir<T>(PREFIX, ni, nj, nk, ni_tranche_size  ) ;

    LOG(info)
        << " ni " << ni
        << " ni_tranche_size " << ni_tranche_size
        << " num_tranche " << num_tranche
        << " reldir " << reldir.c_str()
        << " nj " << nj
        << " nk " << nk
        << " nv(nj*nk) " << nv
        << " size(ni_tranche_size*nv) " << size
        << " typecode " << QU::typecode<T>()
        ;


    // NB seq array memory gets reused for each launch and saved to different paths
    NP* seq = NP::Make<T>(ni_tranche_size, nj, nk) ;
    T* seq_values = seq->values<T>();
    NP::INT seq_nv = seq->num_values();


    LOG(info)
        << " seq " << ( seq ? seq->sstr() : "-" )
        << " seq_values " << seq_values
        << " seq_nv " << seq_nv
        << " seq_values[0] " << seq_values[0]
        << " seq_values[seq_nv-1] " << seq_values[seq_nv-1]
        ;



    for(unsigned t=0 ; t < num_tranche ; t++)
    {
        // *id_offset* controls which rng_state/RNG to use
        unsigned id_offset = ni_tranche_size*t ;
        std::string name = QU::rng_sequence_name<T>(PREFIX, ni_tranche_size, nj, nk, id_offset ) ;

        std::cout
            << std::setw(3) << t
            << std::setw(10) << id_offset
            << std::setw(100) << name.c_str()
            << std::endl
            ;

        rng_sequence( seq_values, ni_tranche_size, nv, id_offset );

        const char* path = spath::Resolve(dir, reldir.c_str(), name.c_str() );
        seq->save(path);
    }
}



template void QSim::rng_sequence<float>(  const char* dir, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size );
template void QSim::rng_sequence<double>( const char* dir, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size );










/**
QSim::scint_wavelength
----------------------------------

Setting envvar QSIM_DISABLE_HD disables multiresolution handling
and causes the returned hd_factor to be zero rather then
the typical values of 10 or 20 which depend on the buffer creation.

**/

extern void QSim_scint_wavelength(   dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, float* wavelength, unsigned num_wavelength );

NP* QSim::scint_wavelength(unsigned num_wavelength, unsigned& hd_factor )
{

    bool qsim_disable_hd = ssys::getenvbool("QSIM_DISABLE_HD");
    hd_factor = qsim_disable_hd ? 0u : scint->tex->getHDFactor() ;
    // HMM: perhaps get this from sim rather than occupying an argument slot
    LOG(LEVEL) << "[" << " qsim_disable_hd " << qsim_disable_hd << " hd_factor " << hd_factor ;

    configureLaunch(num_wavelength, 1 );

    float* d_wavelength = QU::device_alloc<float>(num_wavelength, "QSim::scint_wavelength/num_wavelength");

    QSim_scint_wavelength(numBlocks, threadsPerBlock, d_sim, d_wavelength, num_wavelength );

    NP* w = NP::Make<float>(num_wavelength) ;

    QU::copy_device_to_host_and_free<float>( (float*)w->bytes(), d_wavelength, num_wavelength, "QSim::scint_wavelength" );

    LOG(LEVEL) << "]" ;

    return w ;
}


extern void QSim_RandGaussQ_shoot(  dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, float* v, unsigned num_v );

NP* QSim::RandGaussQ_shoot(unsigned num_v )
{
    const char* label = "QSim::RandGaussQ_shoot/num" ;
    configureLaunch(num_v, 1 );
    std::cout << label << " " << num_v << std::endl;

    float* d_v = QU::device_alloc<float>(num_v, label );

    QSim_RandGaussQ_shoot(numBlocks, threadsPerBlock, d_sim, d_v, num_v );

    cudaDeviceSynchronize();

    NP* v = NP::Make<float>(num_v) ;
    QU::copy_device_to_host_and_free<float>( (float*)v->bytes(), d_v, num_v, label );

    return v ;
}




void QSim::dump_wavelength( float* wavelength, unsigned num_wavelength, unsigned edgeitems )
{
    LOG(LEVEL);
    for(unsigned i=0 ; i < num_wavelength ; i++)
    {
        if( i < edgeitems || i > num_wavelength - edgeitems)
        {
            std::cout
                << std::setw(10) << i
                << std::setw(10) << std::fixed << std::setprecision(3) << wavelength[i]
                << std::endl
                ;
        }
    }
}


extern void QSim_dbg_gs_generate(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, qdebug* dbg, sphoton* photon, unsigned num_photon, unsigned type ) ;


NP* QSim::dbg_gs_generate(unsigned num_photon, unsigned type )
{
    assert( type == SCINT_GENERATE || type == CERENKOV_GENERATE );

    configureLaunch( num_photon, 1 );
    sphoton* d_photon = QU::device_alloc<sphoton>(num_photon, "QSim::dbg_gs_generate:num_photon") ;
    QU::device_memset<sphoton>(d_photon, 0, num_photon);

    QSim_dbg_gs_generate(numBlocks, threadsPerBlock, d_sim, d_dbg, d_photon, num_photon, type );

    NP* p = NP::Make<float>(num_photon, 4, 4);
    const char* label = "QSim::dbg_gs_generate" ;

    QU::copy_device_to_host_and_free<sphoton>( (sphoton*)p->bytes(), d_photon, num_photon, label );
    return p ;
}



extern void QSim_generate_photon(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim )  ;

/**
QSim::generate_photon
-----------------------

**/


void QSim::generate_photon()
{
    LOG(LEVEL) << "[" ;

    unsigned num_photon = qev->getNumPhoton() ;
    LOG(info) << " num_photon " << num_photon ;

    LOG_IF(fatal, num_photon == 0 )
        << " num_photon zero : MUST QEvt::setGenstep before QSim::generate_photon "
        ;

    assert( num_photon > 0 );
    assert( d_sim );

    configureLaunch( num_photon, 1 );

    LOG(info) << "QSim_generate_photon... " ;

    QSim_generate_photon(numBlocks, threadsPerBlock, d_sim );

    LOG(LEVEL) << "]" ;
}






extern void QSim_fill_state_0(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad6* state, unsigned num_state, qdebug* dbg );

void QSim::fill_state_0(quad6* state, unsigned num_state)
{
    assert( d_sim );
    assert( d_dbg );

    quad6* d_state = QU::device_alloc<quad6>(num_state, "QSim::fill_state_0:num_state") ;


    unsigned threads_per_block = 32 ;
    configureLaunch1D( num_state, threads_per_block );

    LOG(info)
         << " num_state " << num_state
         << " threads_per_block  " << threads_per_block
         << " descLaunch " << descLaunch()
         ;

    QSim_fill_state_0(numBlocks, threadsPerBlock, d_sim, d_state, num_state, d_dbg  );

    const char* label = "QSim::fill_state_0" ;
    QU::copy_device_to_host_and_free<quad6>( state, d_state, num_state, label );
}


extern void QSim_fill_state_1(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, sstate* state, unsigned num_state, qdebug* dbg );

void QSim::fill_state_1(sstate* state, unsigned num_state)
{
    assert( d_sim );
    assert( d_dbg );

    sstate* d_state = QU::device_alloc<sstate>(num_state, "QSim::fill_state_1:num_state") ;

    unsigned threads_per_block = 64 ;
    configureLaunch1D( num_state, threads_per_block );

    LOG(info)
         << " num_state " << num_state
         << " threads_per_block  " << threads_per_block
         << " descLaunch " << descLaunch()
         ;

    QSim_fill_state_1(numBlocks, threadsPerBlock, d_sim, d_state, num_state, d_dbg );

    const char* label = "QSim::fill_state_1" ;
    QU::copy_device_to_host_and_free<sstate>( state, d_state, num_state, label );
}








/**
extern QSim_quad_launch
--------------------------

This function is implemented in QSim.cu and it used by *quad_launch_generate*

**/

extern void QSim_quad_launch(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad* q, unsigned num_quad, qdebug* dbg, unsigned type  );



NP* QSim::quad_launch_generate(unsigned num_quad, unsigned type )
{
    assert( d_sim );
    assert( d_dbg );

    const char* label = "QSim::quad_launch_generate:num_quad" ;

    quad* d_q = QU::device_alloc<quad>(num_quad, label ) ;

    unsigned threads_per_block = 512 ;
    configureLaunch1D( num_quad, threads_per_block );

    QSim_quad_launch(numBlocks, threadsPerBlock, d_sim, d_q, num_quad, d_dbg, type );

    NP* q = NP::Make<float>( num_quad, 4 );
    quad* qq = (quad*)q->bytes();

    QU::copy_device_to_host_and_free<quad>( qq, d_q, num_quad, label );

    if( type == QGEN_SMEAR_NORMAL_SIGMA_ALPHA || type == QGEN_SMEAR_NORMAL_POLISH )
    {
        q->set_meta<std::string>("normal", scuda::serialize(dbg->normal) );
        q->set_meta<std::string>("direction", scuda::serialize(dbg->direction) );
        q->set_meta<float>("value", dbg->value );
        q->set_meta<std::string>("valuename", type == QGEN_SMEAR_NORMAL_SIGMA_ALPHA ? "sigma_alpha" : "polish" );
    }

    return q ;
}




/**
extern QSim_photon_launch
--------------------------

This function is implemented in QSim.cu and it used by BOTH *photon_launch_generate* and *photon_launch_mutate*

**/

extern void QSim_photon_launch(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, sphoton* photon, unsigned num_photon, qdebug* dbg, unsigned type  );


/**
QSim::photon_launch_generate
------------------------------

This allocates a photon array on the device, generates photons into it on device and
then downloads the generated photons into the host array. Contrast with *photon_launch_mutate*.

**/

NP* QSim::photon_launch_generate(unsigned num_photon, unsigned type )
{
    assert( d_sim );
    assert( d_dbg );

    const char* label = "QSim::photon_launch_generate:num_photon" ;

    sphoton* d_photon = QU::device_alloc<sphoton>(num_photon, label ) ;
    QU::device_memset<sphoton>(d_photon, 0, num_photon);

    unsigned threads_per_block = 512 ;
    configureLaunch1D( num_photon, threads_per_block );

    QSim_photon_launch(numBlocks, threadsPerBlock, d_sim, d_photon, num_photon, d_dbg, type );

    NP* p = NP::Make<float>(num_photon, 4, 4);
    sphoton* photon = (sphoton*)p->bytes() ;

    QU::copy_device_to_host_and_free<sphoton>( photon, d_photon, num_photon, label );

    return p ;
}




/**
QSim::photon_launch_mutate
---------------------------

This uploads the photon array provided, mutates it and then downloads the changed array.

**/

void QSim::photon_launch_mutate(sphoton* photon, unsigned num_photon, unsigned type )
{
    assert( d_sim );
    assert( d_dbg );

    const char* label_0 = "QSim::photon_launch_mutate/d_photon" ;
    sphoton* d_photon = QU::UploadArray<sphoton>(photon, num_photon, label_0 );

    unsigned DEBUG_NUM_PHOTON = ssys::getenvunsigned(_QSim__photon_launch_mutate_DEBUG_NUM_PHOTON, 0 );
    bool DEBUG_NUM_PHOTON_valid = DEBUG_NUM_PHOTON > 0 && DEBUG_NUM_PHOTON <= num_photon ;
    unsigned u_num_photon = DEBUG_NUM_PHOTON_valid ? DEBUG_NUM_PHOTON  : num_photon ;
    bool SKIP_LAUNCH = ssys::getenvbool(_QSim__photon_launch_mutate_SKIP_LAUNCH) ;

    LOG_IF( error, DEBUG_NUM_PHOTON_valid || true )
        << _QSim__photon_launch_mutate_DEBUG_NUM_PHOTON
        << " DEBUG_NUM_PHOTON " << DEBUG_NUM_PHOTON
        << " num_photon " << num_photon
        << " u_num_photon " << u_num_photon
        << _QSim__photon_launch_mutate_SKIP_LAUNCH
        << " " << ( SKIP_LAUNCH ? "YES" : "NO " )
        ;


    if( SKIP_LAUNCH == false )
    {
        unsigned threads_per_block = 512 ;
        configureLaunch1D( u_num_photon, threads_per_block );
        QSim_photon_launch(numBlocks, threadsPerBlock, d_sim, d_photon, u_num_photon, d_dbg, type );
    }


    const char* label_1 = "QSim::photon_launch_mutate" ;
    QU::copy_device_to_host_and_free<sphoton>( photon, d_photon, u_num_photon, label_1 );
}





/**
QSim::UploadFakePRD (formerly "UploadMockPRD" )
----------------------------------------------------

Caution this returns a device pointer.
**/

quad2* QSim::UploadFakePRD(const NP* ip, const NP* prd) // static
{
    assert(ip);
    int num_ip = ip->shape[0] ;
    assert( num_ip > 0 );

    assert( prd->has_shape( num_ip, -1, 2, 4 ) );    // TODO: evt->max_record checking
    assert( prd->shape.size() == 4 && prd->shape[2] == 2 && prd->shape[3] == 4 );
    int num_prd = prd->shape[0]*prd->shape[1] ;

    LOG(LEVEL)
         << "["
         << " num_ip " << num_ip
         << " num_prd " << num_prd
         << " prd " << prd->sstr()
         ;

    const char* label = "QSim::UploadFakePRD/d_prd" ;
    quad2* d_prd = QU::UploadArray<quad2>( (quad2*)prd->bytes(), num_prd, label );

    // prd is non-standard so it is appropriate to adhoc upload here

    return d_prd ;
}



extern void QSim_fake_propagate_launch(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad2* prd );

/**

QSim::fake_propagate (formerly mock_propagate)
-----------------------------------------------

This is only invoked from QSimTest::fake_propagate


Was renamed from "mock" to "fake" as is within Opticks "mock" is
used to mean without using CUDA.

* number of prd must be a multiple of the number of photon, ratio giving bounce_max
* number of record must be a multiple of the number of photon, ratio giving record_max
* HMM: this is an obtuse way to get bounce_max and record_max

Aiming to replace the above with a simpler and less duplicitous version by
using common QEvt functionality

**/

void QSim::fake_propagate( const NP* prd, unsigned type )
{
    const NP* ip = sev->getInputPhoton();
    int num_ip = ip ? ip->shape[0] : 0 ;
    assert( num_ip > 0 );

    quad2* d_prd = UploadFakePRD(ip, prd) ;

    NP* igs = sev->makeGenstepArrayFromVector();

    int rc = qev->setGenstepUpload_NP(igs);
    assert( rc == 0 );
    if(rc!=0) std::raise(SIGINT);

    sev->add_array("prd0", prd );
    // NB SEvt::beginOfEvent calls SEvt/clear so this addition
    // must be after that to succeed in being added to SEvt saved arrays

    int num_photon = qev->getNumPhoton();
    bool consistent_num_photon = num_photon == num_ip ;

    LOG_IF(fatal, !consistent_num_photon)
         << "["
         << " num_ip " << num_ip
         << " QEvt::getNumPhoton " << num_photon
         << " consistent_num_photon " << ( consistent_num_photon ? "YES" : "NO " )
         << " prd " << prd->sstr()
         ;
    assert(consistent_num_photon);

    assert( qev->upload_count > 0 );

    unsigned threads_per_block = 512 ;
    configureLaunch1D( num_photon, threads_per_block );

    QSim_fake_propagate_launch(numBlocks, threadsPerBlock, d_sim, d_prd );

    cudaDeviceSynchronize();


    LOG(LEVEL) << "]" ;
}



extern void QSim_boundary_lookup_all(    dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, quad* lookup, unsigned width, unsigned height );

NP* QSim::boundary_lookup_all(unsigned width, unsigned height )
{
    LOG(LEVEL) << "[" ;
    assert( bnd );
    assert( width <= getBoundaryTexWidth()  );
    assert( height <= getBoundaryTexHeight()  );

    unsigned num_lookup = width*height ;
    LOG(LEVEL)
        << " width " << width
        << " height " << height
        << " num_lookup " << num_lookup
        ;


    configureLaunch(width, height );

    const char* label = "QSim::boundary_lookup_all:num_lookup" ;

    quad* d_lookup = QU::device_alloc<quad>(num_lookup, label ) ;
    QSim_boundary_lookup_all(numBlocks, threadsPerBlock, d_sim, d_lookup, width, height );

    assert( height % 8 == 0 );
    unsigned num_bnd = height/8 ;

    NP* l = NP::Make<float>( num_bnd, 4, 2, width, 4 );
    QU::copy_device_to_host_and_free<quad>( (quad*)l->bytes(), d_lookup, num_lookup, label );

    LOG(LEVEL) << "]" ;

    return l ;

}

extern void QSim_boundary_lookup_line(    dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, quad* lookup, float* domain, unsigned num_lookup, unsigned line, unsigned k );


NP* QSim::boundary_lookup_line( float* domain, unsigned num_lookup, unsigned line, unsigned k )
{
    LOG(LEVEL)
        << "["
        << " num_lookup " << num_lookup
        << " line " << line
        << " k " << k
        ;

    configureLaunch(num_lookup, 1  );

    float* d_domain = QU::device_alloc<float>(num_lookup, "QSim::boundary_lookup_line:num_lookup") ;

    QU::copy_host_to_device<float>( d_domain, domain, num_lookup );

    const char* label = "QSim::boundary_lookup_line:num_lookup" ;

    quad* d_lookup = QU::device_alloc<quad>(num_lookup, label ) ;

    QSim_boundary_lookup_line(numBlocks, threadsPerBlock, d_sim, d_lookup, d_domain, num_lookup, line, k );


    NP* l = NP::Make<float>( num_lookup, 4 );

    QU::copy_device_to_host_and_free<quad>( (quad*)l->bytes(), d_lookup, num_lookup, label  );

    QU::device_free<float>( d_domain );

    LOG(LEVEL) << "]" ;

    return l ;
}





/**
QSim::prop_lookup
--------------------

suspect problem when have fine domain and many pids due to 2d launch config,
BUT when have 1d launch there is no problem to launch millions of threads : hence the
below *prop_lookup_onebyone*

**/


template <typename T>
extern void QSim_prop_lookup( dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, T* lookup, const T* domain, unsigned domain_width, unsigned* pids, unsigned num_pids );

template <typename T>
void QSim::prop_lookup( T* lookup, const T* domain, unsigned domain_width, const std::vector<unsigned>& pids )
{
    unsigned num_pids = pids.size() ;
    unsigned num_lookup = num_pids*domain_width ;
    LOG(LEVEL)
        << "["
        << " num_pids " << num_pids
        << " domain_width " << domain_width
        << " num_lookup " << num_lookup
        ;

    configureLaunch(domain_width, num_pids  );

    unsigned* d_pids = QU::device_alloc<unsigned>(num_pids, "QSim::prop_lookup:num_pids") ;
    T* d_domain = QU::device_alloc<T>(domain_width, "QSim::prop_lookup:domain_width") ;
    T* d_lookup = QU::device_alloc<T>(num_lookup  , "QSim::prop_lookup:num_lookup") ;

    QU::copy_host_to_device<T>( d_domain, domain, domain_width );
    QU::copy_host_to_device<unsigned>( d_pids, pids.data(), num_pids );

    QSim_prop_lookup(numBlocks, threadsPerBlock, d_sim, d_lookup, d_domain, domain_width, d_pids, num_pids );

    QU::copy_device_to_host_and_free<T>( lookup, d_lookup, num_lookup );
    QU::device_free<T>( d_domain );
    QU::device_free<unsigned>( d_pids );

    LOG(LEVEL) << "]" ;
}



/**
Hmm doing lookups like this is a very common pattern, could do with
a sub context to carry the pieces to simplify doing that.
**/

template <typename T>
extern void QSim_prop_lookup_one(
    dim3 numBlocks,
    dim3 threadsPerBlock,
    qsim* sim,
    T* lookup,
    const T* domain,
    unsigned domain_width,
    unsigned num_pids,
    unsigned pid,
    unsigned ipid
);

/**
QSim::prop_lookup_onebyone
---------------------------

The *lookup* output array holds across domain lookups for multiple property ids.
Separate launches are made for each property id, all writing to the same buffer.

On device uses::

    sim->prop->interpolate

**/

template <typename T>
void QSim::prop_lookup_onebyone( T* lookup, const T* domain, unsigned domain_width, const std::vector<unsigned>& pids )
{
    unsigned num_pids = pids.size() ;
    unsigned num_lookup = num_pids*domain_width ;
    LOG(LEVEL)
        << "["
        << " num_pids " << num_pids
        << " domain_width " << domain_width
        << " num_lookup " << num_lookup
        ;

    configureLaunch(domain_width, 1  );

    T* d_domain = QU::device_alloc<T>(domain_width, "QSim::prop_lookup_onebyone:domain_width") ;
    QU::copy_host_to_device<T>( d_domain, domain, domain_width );

    const char* label = "QSim::prop_lookup_onebyone:num_lookup" ;

    T* d_lookup = QU::device_alloc<T>(num_lookup, label ) ;

    // separate launches for each pid
    for(unsigned ipid=0 ; ipid < num_pids ; ipid++)
    {
        unsigned pid = pids[ipid] ;
        QSim_prop_lookup_one<T>(numBlocks, threadsPerBlock, d_sim, d_lookup, d_domain, domain_width, num_pids, pid, ipid );
    }

    QU::copy_device_to_host_and_free<T>( lookup, d_lookup, num_lookup, label  );

    QU::device_free<T>( d_domain );

    LOG(LEVEL) << "]" ;
}


template void QSim::prop_lookup_onebyone( float*, const float* ,   unsigned, const std::vector<unsigned>& );
template void QSim::prop_lookup_onebyone( double*, const double* , unsigned, const std::vector<unsigned>& );






extern void QSim_multifilm_lookup_all(    dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad2* sample, quad2* result,  unsigned width, unsigned height );

void QSim::multifilm_lookup_all( quad2 * sample , quad2 * result ,  unsigned width, unsigned height )
{
    LOG(LEVEL) << "[" ;
    unsigned num_lookup = width*height ;
    unsigned size = num_lookup ;

    LOG(LEVEL)
        << " width " << width
        << " height " << height
        << " num_lookup " << num_lookup
        << " size "<<size
        ;

    configureLaunch2D(width, height );

    //const float * c_sample = sample;
    quad2* d_sample = QU::device_alloc<quad2>(size, "QSim::multifilm_lookup_all:size" ) ;

    const char* label = "QSim::multifilm_lookup_all:size" ;

    quad2* d_result = QU::device_alloc<quad2>(size, label ) ;
    LOG(LEVEL)
       <<" copy_host_to_device<quad2>( d_sample, sample , size) before";
    QU::copy_host_to_device<quad2>( d_sample, sample , size);
    LOG(LEVEL)
       <<" copy_host_to_device<quad2>( d_sample, sample , size) after";

    QSim_multifilm_lookup_all(numBlocks, threadsPerBlock, d_sim, d_sample, d_result, width, height );
    QU::copy_device_to_host_and_free<quad2>( result , d_result , size, label );
    QU::device_free<quad2>(d_sample);

    cudaDeviceSynchronize();
    LOG(LEVEL) << "]" ;
}




unsigned QSim::getBoundaryTexWidth() const
{
    return bnd->tex->width ;
}
unsigned QSim::getBoundaryTexHeight() const
{
    return bnd->tex->height ;
}
const NP* QSim::getBoundaryTexSrc() const
{
    return bnd->src ;
}

void QSim::dump_photon( quad4* photon, unsigned num_photon, const char* opt_, unsigned edgeitems )
{
    LOG(LEVEL);

    std::string opt = opt_ ;

    bool f0 = opt.find("f0") != std::string::npos ;
    bool f1 = opt.find("f1") != std::string::npos ;
    bool f2 = opt.find("f2") != std::string::npos ;
    bool f3 = opt.find("f3") != std::string::npos ;

    bool i0 = opt.find("i0") != std::string::npos ;
    bool i1 = opt.find("i1") != std::string::npos ;
    bool i2 = opt.find("i2") != std::string::npos ;
    bool i3 = opt.find("i3") != std::string::npos ;

    int wi = 7 ;
    int pr = 2 ;

    for(unsigned i=0 ; i < num_photon ; i++)
    {
        if( i < edgeitems || i > num_photon - edgeitems)
        {
            const quad4& p = photon[i] ;

            std::cout
                << std::setw(wi) << i
                ;

            if(f0) std::cout
                << " f0 "
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q0.f.x
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q0.f.y
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q0.f.z
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q0.f.w
                ;

            if(f1) std::cout
                << " f1 "
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q1.f.x
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q1.f.y
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q1.f.z
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q1.f.w
                ;

            if(f2) std::cout
                << " f2 "
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q2.f.x
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q2.f.y
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q2.f.z
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q2.f.w
                ;

            if(f3) std::cout
                << " f3 "
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q3.f.x
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q3.f.y
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q3.f.z
                << std::setw(wi) << std::fixed << std::setprecision(pr) << p.q3.f.w
                ;

            if(i0) std::cout
                << " i0 "
                << std::setw(wi) << p.q0.i.x
                << std::setw(wi) << p.q0.i.y
                << std::setw(wi) << p.q0.i.z
                << std::setw(wi) << p.q0.i.w
                ;

            if(i1) std::cout
                << " i1 "
                << std::setw(wi) << p.q1.i.x
                << std::setw(wi) << p.q1.i.y
                << std::setw(wi) << p.q1.i.z
                << std::setw(wi) << p.q1.i.w
                ;

            if(i2) std::cout
                << " i2 "
                << std::setw(wi) << p.q2.i.x
                << std::setw(wi) << p.q2.i.y
                << std::setw(wi) << p.q2.i.z
                << std::setw(wi) << p.q2.i.w
                ;

            if(i3) std::cout
                << " i3 "
                << std::setw(wi) << p.q3.i.x
                << std::setw(wi) << p.q3.i.y
                << std::setw(wi) << p.q3.i.z
                << std::setw(wi) << p.q3.i.w
                ;

            std::cout
                << std::endl
                ;
        }
    }
}


/**
QSim::Desc
------------

Generated with::

   ~/opticks/qudarap/QSim__Desc.sh

Dump flags with::

   QSimDescTest
   ssys_test

**/
std::string QSim::Desc(char delim)  // static
{
    std::stringstream ss ;
    ss << ( delim == ',' ? "" : "QSim::Desc\n" )
#ifdef CONFIG_Debug
       << "CONFIG_Debug"
#else
       << "NOT-CONFIG_Debug"
#endif
       << delim
#ifdef CONFIG_RelWithDebInfo
       << "CONFIG_RelWithDebInfo"
#else
       << "NOT-CONFIG_RelWithDebInfo"
#endif
       << delim
#ifdef CONFIG_Release
       << "CONFIG_Release"
#else
       << "NOT-CONFIG_Release"
#endif
       << delim
#ifdef CONFIG_MinSizeRel
       << "CONFIG_MinSizeRel"
#else
       << "NOT-CONFIG_MinSizeRel"
#endif
       << delim
#ifdef PRODUCTION
       << "PRODUCTION"
#else
       << "NOT-PRODUCTION"
#endif
       << delim
#ifdef WITH_CHILD
       << "WITH_CHILD"
#else
       << "NOT-WITH_CHILD"
#endif
       << delim
#ifdef WITH_CUSTOM4
       << "WITH_CUSTOM4"
#else
       << "NOT-WITH_CUSTOM4"
#endif
       << delim
#ifdef PLOG_LOCAL
       << "PLOG_LOCAL"
#else
       << "NOT-PLOG_LOCAL"
#endif
       << delim
#ifdef DEBUG_PIDX
       << "DEBUG_PIDX"
#else
       << "NOT-DEBUG_PIDX"
#endif
       << delim
#ifdef DEBUG_TAG
       << "DEBUG_TAG"
#else
       << "NOT-DEBUG_TAG"
#endif
       << delim
#ifdef RNG_XORWOW
       << "RNG_XORWOW"
#else
       << "NOT-RNG_XORWOW"
#endif
       << delim
#ifdef RNG_PHILOX
       << "RNG_PHILOX"
#else
       << "NOT-RNG_PHILOX"
#endif
       << delim
#ifdef RNG_PHILITEOX
       << "RNG_PHILITEOX"
#else
       << "NOT-RNG_PHILITEOX"
#endif
       << delim
       ;
    std::string str = ss.str() ;
    return str ;
}



std::string QSim::Switches()  // static
{
    return Desc(',');
}


