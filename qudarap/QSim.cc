#include "PLOG.hh"
#include "SSys.hh"
#include "SPath.hh"
#include "scuda.h"
#include "squad.h"

#include "NP.hh"
#include "QUDA_CHECK.h"
#include "QU.hh"

#include "qrng.h"
#include "qsim.h"
#include "qdebug.h"

#include "QEvent.hh"
#include "QRng.hh"
#include "QTex.hh"
#include "QScint.hh"
#include "QBnd.hh"
#include "QPrd.hh"
#include "QProp.hh"
#include "QMultiFilmLUT.hh"
#include "QEvent.hh"
#include "QOptical.hh"
#include "QState.hh"
#include "QSimLaunch.hh"

#include "QSim.hh"

const plog::Severity QSim::LEVEL = PLOG::EnvLevel("QSim", "INFO"); 

const QSim* QSim::INSTANCE = nullptr ; 

const QSim* QSim::Get(){ return INSTANCE ; }

/**
QSim::UploadComponents
-----------------------

This is invoked for example by CSGOptiX/tests/CSGOptiXSimtraceTest.cc 
prior to instanciating CSGOptiX 

Uploading components is a once only action for a geometry, encompassing:

* random states
* scintillation textures 
* boundary textures
* property arrays

It is the simulation physics equivalent of uploading the CSGFoundry geometry. 

The components are manages by separate singleton instances 
that subsequent QSim instanciation collects together.
This structure is used to allow separate testing. 


TODO: Most of the component arguments come from CSGFoundry but it is not possible
to consolidate to just CSGFoundry argument as that would add CSG dependency to QUDARap
which is not acceptable. PERHAPS: accept single argument "std::map<std::string, const NP*>&" 
argument with meaningful standardized keys. 

**/

void QSim::UploadComponents( const NP* icdf_, const NP* bnd, const NP* optical, const char* rindexpath  )
{
    QRng* qrng = new QRng ;  // loads and uploads curandState 
    LOG(LEVEL) << qrng->desc(); 

    const char* qsim_icdf_path = SSys::getenvvar("QSIM_ICDF_PATH", nullptr ); 
    const NP* override_icdf = qsim_icdf_path ?  NP::Load(qsim_icdf_path) : nullptr ;
    const NP* icdf = override_icdf ? override_icdf : icdf_ ; 
 
    if( icdf == nullptr )
    {
        LOG(warning) << " icdf null " ; 
    }
    else
    {
        unsigned hd_factor = 20u ;  // 0,10,20
        QScint* qscint = new QScint( icdf, hd_factor); // custom high-definition inverse CDF for scintillation generation
        LOG(LEVEL) << qscint->desc(); 
    }


    // TODO: combine QOptical and QBnd as very closely related 

    if( bnd == nullptr )
    {
        LOG(warning) << " bnd null " ; 
    }
    else
    {
        QBnd* qbnd = new QBnd(bnd); // boundary texture with standard domain, used for standard fast property lookup 
        LOG(LEVEL) << qbnd->desc(); 
    }

    if( optical == nullptr )
    {
        LOG(warning) << " optical null " ; 
    }
    else
    {
        QOptical* qopt = new QOptical(optical); 
        LOG(fatal) << qopt->desc(); 
    }





    LOG(error) << "[ QProp " ; 
    QProp<float>* qprop = new QProp<float>(rindexpath) ;  // property interpolation with per-property domains, eg used for Cerenkov RINDEX sampling 
    LOG(error) << "] QProp " ; 


    LOG(LEVEL) << qprop->desc(); 
}

/**
QSim::UploadMultiFilmLUT
--------------------------
  
 instance QMultiFilmLUT and upload the component : lookup table

**/

void QSim::UploadMultiFilmLUT( const NP* multi_film_lut ){

    if( multi_film_lut == nullptr )
    {
        LOG(warning) << " multi_film null ";
    }
    else
    {
        QMultiFilmLUT* qmul = new QMultiFilmLUT( multi_film_lut ); 
        LOG(LEVEL) << qmul->desc();
    
    }
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
    event(new QEvent),
    rng(QRng::Get()),
    scint(QScint::Get()),
    bnd(QBnd::Get()),
    prd(QPrd::Get()),
    optical(QOptical::Get()),
    prop(QProp<float>::Get()),
    multi_film(QMultiFilmLUT::Get()),
    pidx(SSys::getenvint("PIDX", -1)),
    sim(nullptr),
    d_sim(nullptr),
    dbg(nullptr), 
    d_dbg(nullptr)
{
    init(); 
}

void QSim::init()
{
    LOG(LEVEL) << event->descMax(); 

    init_sim(); 
    init_dbg(); 

    INSTANCE = this ; 
    LOG(LEVEL) << desc() ; 
}

/**
QSim::init_sim
--------------------

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

In a very real sense it is object oriented GPU launches. 

**/

void QSim::init_sim()
{
    sim = new qsim ; 

    LOG(LEVEL) 
        << " rng " << rng 
        << " scint " << scint
        << " bnd " << bnd
        << " optical " << optical
        << " prop " << prop
        << " pidx " << pidx
        << " sim " << sim 
        << " d_sim " << d_sim 
        ;  

    if(event)
    {
        LOG(LEVEL) << " event " << event->desc() ; 
        sim->evt = event->d_evt ; 
    }

    if(rng)
    {
        LOG(LEVEL) << " rng " << rng->desc() ; 
        sim->rngstate = rng->qr->rng_states ; 
    } 
    if(scint)
    {
        /**

        unsigned hd_factor = scint->tex->getHDFactor() ;  // HMM: perhaps get this from sim rather than occupying an argument slot  
        LOG(LEVEL) 
            << " scint.desc " << scint->desc() 
            << " hd_factor " << hd_factor 
            ;

        sim->scint_tex = scint->tex->texObj ; 
        sim->scint_meta = scint->tex->d_meta ; 

        **/
        sim->scint = scint->d_scint ; 

    } 
    if(bnd)
    {
        LOG(LEVEL) << " bnd " << bnd->desc() ; 
        sim->boundary_tex = bnd->tex->texObj ; 
        sim->boundary_meta = bnd->tex->d_meta ; 

        assert( sim->boundary_meta != nullptr ); 

        sim->boundary_tex_MaterialLine_Water = bnd->getMaterialLine("Water") ; 
        sim->boundary_tex_MaterialLine_LS    = bnd->getMaterialLine("LS") ; 
    } 
    if(optical)
    {
        LOG(LEVEL) << " optical " << optical->desc() ; 
        sim->optical = optical->d_optical ; 
    }

    if(prop)
    {
        LOG(LEVEL) << " prop " << prop->desc() ; 
        sim->prop = prop->getDevicePtr() ; 
    }

    if(multi_film)
    {
        sim->qmultifilm = multi_film->getDevicePtr();
    }   

    sim->pidx = pidx ; 

    d_sim = QU::UploadArray<qsim>(sim, 1 );  
}


/**
QSim::init_db
----------------

*dbg* is a host side instance that is populated by this method and 
then uploaded to the device *d_dbg* 

qdebug avoids having to play pass the parameter thru multiple levels of calls  
to get values onto the device 

Notice how not using pointers in qdebug provides a simple plain old struct way 
to get structured info onto the device. 

**/

void QSim::init_dbg()
{
    dbg = new qdebug ; 

    // miscellaneous used by fill_state testing 

    float cosTheta = 0.5f ; 
    dbg->wavelength = 500.f ; 
    dbg->cosTheta = cosTheta ; 
    qvals( dbg->normal , "DBG_NRM", "0,0,1" ); 
   
    // qstate: mocking result of fill_state 
    dbg->s = QState::Make(); 
    LOG(info) << desc_dbg_state(); 

    // quad2: mocking prd per-ray-data result of optix trace calls 
    dbg->prd = quad2::make_eprd() ;  // see qudarap/tests/eprd.sh 
     
    dbg->p.ephoton() ; // see qudarap/tests/ephoton.sh 
    LOG(info) << desc_dbg_p0()  ; 

    d_dbg = QU::UploadArray<qdebug>(dbg, 1 );  
}


NP* QSim::duplicate_dbg_ephoton(unsigned num_photon)
{
    NP* ph = NP::Make<float>(num_photon, 4, 4 );
    sphoton* pp  = (sphoton*)ph->bytes(); 
    for(unsigned i=0 ; i < num_photon ; i++)
    {   
        sphoton p = dbg->p  ;  // start from ephoton 
        p.pos.y = float(i)*100.f ; 
        pp[i] = p ;   
    }    
    return ph ; 
}


std::string QSim::desc_dbg_state() const 
{
    std::stringstream ss ; 
    ss << "QSim::desc_dbg_state" << std::endl << QState::Desc(dbg->s) ; 
    std::string s = ss.str(); 
    return s ; 
}
 
std::string QSim::desc_dbg_p0() const 
{
    std::stringstream ss ; 
    ss << "QSim::desc_dbg_p0" << std::endl << dbg->p.desc() ; 
    std::string s = ss.str(); 
    return s ; 
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
    ss << "QSim"
       << " sim->rngstate " << sim->rngstate 
       << " sim->scint " << sim->scint 
       << " sim->boundary_tex " << sim->boundary_tex 
       << " sim->boundary_meta " << sim->boundary_meta
       << " d_sim " << d_sim 
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
extern void QSim_rng_sequence(dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, T* seq, unsigned ni, unsigned nj, unsigned id_offset ); 


/**
QSim::rng_sequence generate randoms in single CUDA launch
-------------------------------------------------------------

This is invoked for each tranche by the below rng_sequence method.
Each tranche launch generates ni_tranche*nv randoms writing them into seq

ni_tranche : item tranche size

nv : number randoms to generate for each item

id_offset : acts on the rngstates array 

**/

template <typename T>
void QSim::rng_sequence( T* seq, unsigned ni_tranche, unsigned nv, unsigned id_offset )
{
    configureLaunch(ni_tranche, 1 ); 

    unsigned num_rng = ni_tranche*nv ;  

    T* d_seq = QU::device_alloc<T>(num_rng); 

    QSim_rng_sequence<T>(numBlocks, threadsPerBlock, d_sim, d_seq, ni_tranche, nv, id_offset );  

    QU::copy_device_to_host_and_free<T>( seq, d_seq, num_rng ); 
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

**/

template <typename T>
void QSim::rng_sequence( const char* dir, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size  )
{
    assert( ni >= ni_tranche_size && ni % ni_tranche_size == 0 );   // total size *ni* must be integral multiple of *ni_tranche_size*
    unsigned num_tranche = ni/ni_tranche_size ; 
    unsigned nv = nj*nk ; 

    unsigned size = ni_tranche_size*nv ;   // number of randoms to be generated in each launch 
    std::string reldir = QU::rng_sequence_reldir<T>(PREFIX, ni, nj, nk, ni_tranche_size  ) ;  

    std::cout 
        << "QSim::rng_sequence" 
        << " ni " << ni
        << " ni_tranche_size " << ni_tranche_size
        << " num_tranche " << num_tranche 
        << " reldir " << reldir.c_str()
        << " nj " << nj
        << " nk " << nk
        << " nv(nj*nk) " << nv 
        << " size(ni_tranche_size*nv) " << size 
        << " typecode " << QU::typecode<T>() 
        << std::endl 
        ; 


    // NB seq array memory gets reused for each launch and saved to different paths
    NP* seq = NP::Make<T>(ni_tranche_size, nj, nk) ; 
    T* values = seq->values<T>(); 

    for(unsigned t=0 ; t < num_tranche ; t++)
    {
        // *id_offset* controls which rngstates/curandState to use
        unsigned id_offset = ni_tranche_size*t ; 
        std::string name = QU::rng_sequence_name<T>(PREFIX, ni_tranche_size, nj, nk, id_offset ) ;  

        std::cout 
            << std::setw(3) << t 
            << std::setw(10) << id_offset 
            << std::setw(100) << name.c_str()
            << std::endl 
            ; 

        rng_sequence( values, ni_tranche_size, nv, id_offset );  
         
        int create_dirs = 1 ;  // 1:filepath
        const char* path = SPath::Resolve(dir, reldir.c_str(), name.c_str(), create_dirs ); 

        seq->save(path); 
    }
}



template void QSim::rng_sequence<float>(  const char* dir, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size  ); 
template void QSim::rng_sequence<double>( const char* dir, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size  ); 



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

    bool qsim_disable_hd = SSys::getenvbool("QSIM_DISABLE_HD"); 
    hd_factor = qsim_disable_hd ? 0u : scint->tex->getHDFactor() ; 
    // HMM: perhaps get this from sim rather than occupying an argument slot  
    LOG(LEVEL) << "[" << " qsim_disable_hd " << qsim_disable_hd << " hd_factor " << hd_factor ; 

    configureLaunch(num_wavelength, 1 ); 

    float* d_wavelength = QU::device_alloc<float>(num_wavelength); 

    QSim_scint_wavelength(numBlocks, threadsPerBlock, d_sim, d_wavelength, num_wavelength );  

    NP* w = NP::Make<float>(num_wavelength) ; 

    QU::copy_device_to_host_and_free<float>( (float*)w->bytes(), d_wavelength, num_wavelength ); 

    LOG(LEVEL) << "]" ; 

    return w ; 
}

/**
QSim::cerenkov_wavelength
---------------------------

**/

extern void QSim_cerenkov_wavelength_rejection_sampled(dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, float* wavelength, unsigned num_wavelength ); 

NP* QSim::cerenkov_wavelength_rejection_sampled(unsigned num_wavelength )
{
    LOG(LEVEL) << "[ num_wavelength " << num_wavelength ;
 
    configureLaunch(num_wavelength, 1 ); 

    float* d_wavelength = QU::device_alloc<float>(num_wavelength); 

    QSim_cerenkov_wavelength_rejection_sampled(numBlocks, threadsPerBlock, d_sim, d_wavelength, num_wavelength );  

    NP* w = NP::Make<float>(num_wavelength) ; 

    QU::copy_device_to_host_and_free<float>( (float*)w->bytes(), d_wavelength, num_wavelength ); 

    LOG(LEVEL) << "]" ; 

    return w ; 
}




extern void QSim_cerenkov_photon(dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, quad4* photon, unsigned num_photon );

template <typename T>
extern void QSim_cerenkov_photon_enprop(dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, quad4* photon, unsigned num_photon );

extern void QSim_cerenkov_photon_expt(dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, quad4* photon, unsigned num_photon );



NP* QSim::cerenkov_photon(unsigned num_photon, unsigned test )
{
    configureLaunch(num_photon, 1 ); 
    quad4* d_photon = QU::device_alloc<quad4>(num_photon); 

    switch(test)
    {
        case CERENKOV_PHOTON:                QSim_cerenkov_photon(              numBlocks,  threadsPerBlock, d_sim, d_photon, num_photon ); break ; 
        case CERENKOV_PHOTON_ENPROP_FLOAT:   QSim_cerenkov_photon_enprop<float>(numBlocks,  threadsPerBlock, d_sim, d_photon, num_photon ); break ; 
        case CERENKOV_PHOTON_ENPROP_DOUBLE:  QSim_cerenkov_photon_enprop<double>(numBlocks, threadsPerBlock, d_sim, d_photon, num_photon ); break ; 
        case CERENKOV_PHOTON_EXPT:           QSim_cerenkov_photon_expt(          numBlocks, threadsPerBlock, d_sim, d_photon, num_photon ); break ; 
    }

    NP* p = NP::Make<float>( num_photon, 4, 4); 
    quad4* pp = (quad4*)p->bytes() ; 
    QU::copy_device_to_host_and_free<quad4>( pp, d_photon, num_photon ); 
    return p ; 
}


/*
void QSim::cerenkov_photon( quad4* photon, unsigned num_photon )
{
    configureLaunch(num_photon, 1 ); 
    quad4* d_photon = QU::device_alloc<quad4>(num_photon); 
    QSim_cerenkov_photon(numBlocks, threadsPerBlock, d_sim, d_photon, num_photon );  
    QU::copy_device_to_host_and_free<quad4>( photon, d_photon, num_photon ); 
}

template<typename T>
void QSim::cerenkov_photon_enprop( quad4* photon, unsigned num_photon )
{
    configureLaunch(num_photon, 1 ); 
    quad4* d_photon = QU::device_alloc<quad4>(num_photon); 
    QSim_cerenkov_photon_enprop<T>(numBlocks, threadsPerBlock, d_sim, d_photon, num_photon );  
    QU::copy_device_to_host_and_free<quad4>( photon, d_photon, num_photon ); 
}

template void QSim::cerenkov_photon_enprop<float>(  quad4*, unsigned ); 
template void QSim::cerenkov_photon_enprop<double>( quad4*, unsigned ); 

void QSim::cerenkov_photon_expt( quad4* photon, unsigned num_photon )
{
    configureLaunch(num_photon, 1 ); 
    quad4* d_photon = QU::device_alloc<quad4>(num_photon); 
    QSim_cerenkov_photon_expt(numBlocks, threadsPerBlock, d_sim, d_photon, num_photon );  
    QU::copy_device_to_host_and_free<quad4>( photon, d_photon, num_photon ); 
}
*/




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


extern void QSim_scint_photon( dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, quad4* photon , unsigned num_photon ); 


NP* QSim::scint_photon(unsigned num_photon )
{
    configureLaunch( num_photon, 1 ); 
    quad4* d_photon = QU::device_alloc<quad4>(num_photon) ; 
    QSim_scint_photon(numBlocks, threadsPerBlock, d_sim, d_photon, num_photon );  
    NP* p = NP::Make<float>(num_photon, 4, 4); 
    QU::copy_device_to_host_and_free<quad4>( (quad4*)p->bytes(), d_photon, num_photon ); 
    return p ; 
}




extern void QSim_generate_photon(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, qevent* evt )  ; 
 
/**
QSim::generate_photon
-----------------------

**/


void QSim::generate_photon(QEvent* evt)
{
    LOG(LEVEL) << "[" ; 

    unsigned num_photon = evt->getNumPhoton() ;  
    LOG(info) << " num_photon " << num_photon ; 

    if( num_photon == 0 )
    {
        LOG(fatal) 
           << " num_photon zero : MUST QEvent::setGenstep before QSim::generate_photon "  
           ; 
        return ; 
    }

    qevent* d_evt = evt->getDevicePtr(); 

    assert( d_evt ); 
    assert( d_sim ); 

    configureLaunch( num_photon, 1 ); 

    LOG(info) << "QSim_generate_photon... " ; 

    QSim_generate_photon(numBlocks, threadsPerBlock, d_sim, d_evt );  

    LOG(LEVEL) << "]" ; 
}






extern void QSim_fill_state_0(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad6* state, unsigned num_state, qdebug* dbg ); 

void QSim::fill_state_0(quad6* state, unsigned num_state)
{
    assert( d_sim ); 
    assert( d_dbg ); 

    quad6* d_state = QU::device_alloc<quad6>(num_state) ; 


    unsigned threads_per_block = 32 ;  
    configureLaunch1D( num_state, threads_per_block ); 

    LOG(info) 
         << " num_state " << num_state 
         << " threads_per_block  " << threads_per_block
         << " descLaunch " << descLaunch()
         ; 

    QSim_fill_state_0(numBlocks, threadsPerBlock, d_sim, d_state, num_state, d_dbg  );  

    QU::copy_device_to_host_and_free<quad6>( state, d_state, num_state ); 
}


extern void QSim_fill_state_1(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, qstate* state, unsigned num_state, qdebug* dbg ); 

void QSim::fill_state_1(qstate* state, unsigned num_state)
{
    assert( d_sim ); 
    assert( d_dbg ); 

    qstate* d_state = QU::device_alloc<qstate>(num_state) ; 

    unsigned threads_per_block = 64 ;  
    configureLaunch1D( num_state, threads_per_block ); 

    LOG(info) 
         << " num_state " << num_state 
         << " threads_per_block  " << threads_per_block
         << " descLaunch " << descLaunch()
         ; 

    QSim_fill_state_1(numBlocks, threadsPerBlock, d_sim, d_state, num_state, d_dbg );  

    QU::copy_device_to_host_and_free<qstate>( state, d_state, num_state ); 
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

    quad* d_q = QU::device_alloc<quad>(num_quad) ; 

    unsigned threads_per_block = 512 ;  
    configureLaunch1D( num_quad, threads_per_block ); 

    QSim_quad_launch(numBlocks, threadsPerBlock, d_sim, d_q, num_quad, d_dbg, type );  

    NP* q = NP::Make<float>( num_quad, 4 ); 
    quad* qq = (quad*)q->bytes(); 

    QU::copy_device_to_host_and_free<quad>( qq, d_q, num_quad ); 

    return q ; 
}
 


/**
extern QSim_photon_launch
--------------------------

This function is implemented in QSim.cu and it used by *photon_launch_generate* and *photon_launch_mutate* 

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

    sphoton* d_photon = QU::device_alloc<sphoton>(num_photon) ; 
    unsigned threads_per_block = 512 ;  
    configureLaunch1D( num_photon, threads_per_block ); 

    NP* p = NP::Make<float>(num_photon, 4, 4); 
    sphoton* photon = (sphoton*)p->bytes() ; 
    QU::copy_device_to_host_and_free<sphoton>( photon, d_photon, num_photon ); 
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

    sphoton* d_photon = QU::UploadArray<sphoton>(photon, num_photon );  

    unsigned threads_per_block = 512 ;  
    configureLaunch1D( num_photon, threads_per_block ); 

    QSim_photon_launch(numBlocks, threadsPerBlock, d_sim, d_photon, num_photon, d_dbg, type );  

    QU::copy_device_to_host_and_free<sphoton>( photon, d_photon, num_photon ); 
}
 
 

extern void QSim_mock_propagate_launch(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad2* prd );  


/**
QSim::mock_propagate_launch_mutate
------------------------------------

* number of prd must be a multiple of the number of photon, ratio giving bounce_max 
* number of record must be a multiple of the number of photon, ratio giving record_max 
* HMM: this is an obtuse way to get bounce_max and record_max 

Aiming to replace the above with a simpler and less duplicitous version by 
using common QEvent functionality 

**/

void QSim::mock_propagate( NP* p, const NP* prd, unsigned type )
{
    int num_p = p->shape[0] ; 

    assert( prd->has_shape( num_p, -1, 2, 4 ) );    // TODO: evt->max_record checking 
    assert( prd->shape.size() == 4 && prd->shape[2] == 2 && prd->shape[3] == 4 ); 

    int num_prd = prd->shape[0]*prd->shape[1] ;  

    LOG(info) << "[ num_prd " << num_prd << " prd " << prd->sstr()  ;
 
    event->setPhoton(p); 

    int num_photon = event->evt->num_photon ; 
    assert( num_photon == num_p ); 

    quad2* d_prd = QU::UploadArray<quad2>( (quad2*)prd->bytes(), num_prd );  
    // prd non-standard so appropriate to upload here 

    unsigned threads_per_block = 512 ;  
    configureLaunch1D( num_photon, threads_per_block ); 

    QSim_mock_propagate_launch(numBlocks, threadsPerBlock, d_sim, d_prd );  

    cudaDeviceSynchronize();

    event->getPhoton(p); 
    LOG(info) << "]" ; 
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

    quad* d_lookup = QU::device_alloc<quad>(num_lookup) ; 
    QSim_boundary_lookup_all(numBlocks, threadsPerBlock, d_sim, d_lookup, width, height );  

    assert( height % 8 == 0 );  
    unsigned num_bnd = height/8 ;   

    NP* l = NP::Make<float>( num_bnd, 4, 2, width, 4 ); 
    QU::copy_device_to_host_and_free<quad>( (quad*)l->bytes(), d_lookup, num_lookup ); 

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

    float* d_domain = QU::device_alloc<float>(num_lookup) ; 

    QU::copy_host_to_device<float>( d_domain, domain, num_lookup ); 

    quad* d_lookup = QU::device_alloc<quad>(num_lookup) ; 

    QSim_boundary_lookup_line(numBlocks, threadsPerBlock, d_sim, d_lookup, d_domain, num_lookup, line, k );  


    NP* l = NP::Make<float>( num_lookup, 4 ); 

    QU::copy_device_to_host_and_free<quad>( (quad*)l->bytes(), d_lookup, num_lookup ); 

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

    unsigned* d_pids = QU::device_alloc<unsigned>(num_pids) ; 
    T* d_domain = QU::device_alloc<T>(domain_width) ; 
    T* d_lookup = QU::device_alloc<T>(num_lookup) ; 

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

    T* d_domain = QU::device_alloc<T>(domain_width) ; 
    QU::copy_host_to_device<T>( d_domain, domain, domain_width ); 

    T* d_lookup = QU::device_alloc<T>(num_lookup) ; 

    // separate launches for each pid
    for(unsigned ipid=0 ; ipid < num_pids ; ipid++)
    {
        unsigned pid = pids[ipid] ; 
        QSim_prop_lookup_one<T>(numBlocks, threadsPerBlock, d_sim, d_lookup, d_domain, domain_width, num_pids, pid, ipid );  
    }

    QU::copy_device_to_host_and_free<T>( lookup, d_lookup, num_lookup ); 

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
    quad2* d_sample = QU::device_alloc<quad2>(size) ;
    
    quad2* d_result = QU::device_alloc<quad2>(size) ;
    LOG(LEVEL)
       <<" copy_host_to_device<quad2>( d_sample, sample , size) before";
    QU::copy_host_to_device<quad2>( d_sample, sample , size);
    LOG(LEVEL)
       <<" copy_host_to_device<quad2>( d_sample, sample , size) after";

    QSim_multifilm_lookup_all(numBlocks, threadsPerBlock, d_sim, d_sample, d_result, width, height );  
    QU::copy_device_to_host_and_free<quad2>( result , d_result , size); 
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


