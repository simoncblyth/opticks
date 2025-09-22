#pragma once

#include <string>
#include <vector>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

/**
QSim
======

The canonical QSim instance is instanciated with CSGOptiX::CSGOptiX

QSim is mostly constant and needs initializing once only, as
it corresponds to physics and fixed parameters of the detector,
making it analogous to the CSGFoundry geometry.

Contrast with the QEvent with a very different event-by-event lifecycle

HMM : MOST OF THIS API IS FOR TESTING ONLY  : TODO: Move lots to QSimTest perhaps ?

**/

struct NP ;
struct SSim ;
struct SEvt ;

template <typename T> struct QTex ;
template <typename T> struct QBuf ;
template <typename T> struct QProp ;
template <typename T> struct QPMT ;

struct qsim ;

struct QBase ;
struct QEvent ;
struct QRng ;
struct QScint ;
struct QCerenkov ;
struct QBnd ;
struct QMultiFilm;
struct QOptical ;
struct QEvent ;
struct QDebug ;

struct qdebug ;
struct sstate ;

struct quad4 ;
struct quad2 ;
struct sphoton ;
union  quad ;

struct SCSGOptiX ;

struct QUDARAP_API QSim
{
    static constexpr const int M = 1000000 ;
    static const plog::Severity LEVEL ;
    static const char* PREFIX ;
    static QSim* INSTANCE ;
    static QSim* Get();
    static QSim* Create();

    static void UploadComponents(const SSim* ssim);

    const QBase*     base ;
    QEvent*          event ;
    SEvt*            sev ;

    const QRng*      rng ;
    const QScint*    scint ;
    const QCerenkov* cerenkov ;
    const QBnd*      bnd ;
    const QOptical*  optical ;
    const QDebug*    debug_ ;

    const QProp<float>*  prop ;
    const QPMT<float>*   pmt ;
    const QMultiFilm*    multifilm ;

    qsim*                 sim ;
    qsim*               d_sim ;

    qdebug*           dbg ;
    qdebug*           d_dbg ;

    SCSGOptiX*        cx ;


    dim3 numBlocks ;
    dim3 threadsPerBlock ;

private:
    QSim();
    void init();

    static constexpr const char* _QSim__REQUIRE_PMT = "QSim__REQUIRE_PMT" ;
    static const bool   REQUIRE_PMT;

    static constexpr const char* _QSim__SAVE_IGS_EVENTID = "QSim__SAVE_IGS_EVENTID" ;
    static const int   SAVE_IGS_EVENTID ;

    static constexpr const char* _QSim__SAVE_IGS_PATH    = "QSim__SAVE_IGS_PATH" ;
    static const char* SAVE_IGS_PATH ;

    static constexpr const char* _QSim__CONCAT    = "QSim__CONCAT" ;
    static const bool CONCAT ;

    static constexpr const char* _QSim__ALLOC    = "QSim__ALLOC" ;
    static const bool ALLOC ;


public:
    void setLauncher(SCSGOptiX* cx_ );

    static constexpr const char* QSim__simulate_KEEP_SUBFOLD = "QSim__simulate_KEEP_SUBFOLD" ;
    static bool KEEP_SUBFOLD ;

    double simulate(int eventID, bool reset_ );      // via cx launch
    NP*    simulate(const NP* gs);                   // higher level API for use from CSGOptiXService.h

    static void MaybeSaveIGS(int eventID, NP* igs);

    int    getPhotonSlotOffset() const ;

    void   reset( int eventID);

    double simtrace(int eventID);


    qsim* getDevicePtr() const ;
    std::string desc() const ;
    std::string descFull() const ;
    std::string descComponents() const ;


    // TODO: relocate non-essential methods into tests or elsewhere

    char getScintTexFilterMode() const ;

    void configureLaunch16();
    void configureLaunch( unsigned width, unsigned height );
    void configureLaunch2D( unsigned width, unsigned height );
    void configureLaunch1D(unsigned num, unsigned threads_per_block);
    std::string descLaunch() const ;


    template<typename T>
    void rng_sequence( dim3 numblocks, dim3 threadsPerBlock, qsim* d_sim, T* d_seq, unsigned ni_tranche, unsigned nv, unsigned ioffset );

    template<typename T>
    void rng_sequence( T* seq, unsigned ni, unsigned nj, unsigned ioffset );

    template<typename T>
    void rng_sequence( const char* dir, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size );


    NP* scint_wavelength( unsigned num_wavelength, unsigned& hd_factor );

    NP* RandGaussQ_shoot(unsigned num_v );


    // NP* cerenkov_wavelength_rejection_sampled( unsigned num_wavelength );
    void dump_wavelength(                       float* wavelength, unsigned num_wavelength, unsigned edgeitems=10 );


    NP* dbg_gs_generate(unsigned num_photon, unsigned type );


    void dump_photon(            quad4* photon, unsigned num_photon, const char* opt="f0,f1,f2,i3", unsigned egdeitems=10 );

    void generate_photon();
    void fill_state_0(quad6*  state, unsigned num_state);
    void fill_state_1(sstate* state, unsigned num_state);

    NP* quad_launch_generate(unsigned num_quad, unsigned type );
    NP* photon_launch_generate(unsigned num_photon, unsigned type );

    static constexpr const char* _QSim__photon_launch_mutate_DEBUG_NUM_PHOTON = "QSim__photon_launch_mutate_DEBUG_NUM_PHOTON" ;
    static constexpr const char* _QSim__photon_launch_mutate_SKIP_LAUNCH = "QSim__photon_launch_mutate_SKIP_LAUNCH" ;
    void photon_launch_mutate(   sphoton* photon, unsigned num_photon, unsigned type );


    static quad2* UploadFakePRD(const NP* ip, const NP* prd);
    void fake_propagate(const NP* prd, unsigned type );

    unsigned getBoundaryTexWidth() const ;
    unsigned getBoundaryTexHeight() const ;
    const NP* getBoundaryTexSrc() const ;

    NP* boundary_lookup_all( unsigned width, unsigned height ) ;
    NP* boundary_lookup_line( float* domain, unsigned num_lookup, unsigned line, unsigned k ) ;


    template<typename T>
    void prop_lookup(          T* lookup, const T* domain, unsigned domain_width, const std::vector<unsigned>& pids ) ;

    template<typename T>
    void prop_lookup_onebyone( T* lookup, const T* domain, unsigned domain_width, const std::vector<unsigned>& pids ) ;

    void multifilm_lookup_all( quad2* sample , quad2* result ,  unsigned width, unsigned height );

    static std::string Desc(char delim='\n');
    static std::string Switches();
};


