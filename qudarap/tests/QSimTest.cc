#include <sstream>

#include <cuda_runtime.h>
#include "OPTICKS_LOG.hh"

#ifdef OLD
#include "Opticks.hh"
#else
#include "SOpticksResource.hh"
#endif

#include "scuda.h"
#include "squad.h"
#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"

#include "QRng.hh"
#include "QBnd.hh"
#include "QProp.hh"
#include "QSim.hh"
#include "QEvent.hh"
#include "QDebug.hh"

#include "qstate.h"

#include "SEvent.hh"


enum { 
   UNKNOWN,
   RNG_SEQUENCE_F,
   WAVELENGTH_S,
   WAVELENGTH_C,
   SCINT_PHOTON_P,
   CERENKOV_PHOTON_K,
   CERENKOV_PHOTON_ENPROP_E,
   CERENKOV_PHOTON_EXPT_X,
   GENERATE_PHOTON_G,
   BOUNDARY_LOOKUP_ALL_A,
   BOUNDARY_LOOKUP_LINE_WATER_W,
   BOUNDARY_LOOKUP_LINE_LS_L,
   PROP_LOOKUP_Y,
   FILL_STATE_0,
   FILL_STATE_1,
   PROPAGATE_TO_BOUNDARY,
   RAYLEIGH_SCATTER_ALIGN
} ;
 
unsigned TestType( const char* name )
{
   unsigned test = UNKNOWN ;  
   if(strcmp(name,"F") == 0 ) test = RNG_SEQUENCE_F ; 
   if(strcmp(name,"S") == 0 ) test = WAVELENGTH_S ; 
   if(strcmp(name,"C") == 0 ) test = WAVELENGTH_C ;
   if(strcmp(name,"P") == 0 ) test = SCINT_PHOTON_P ;
   if(strcmp(name,"K") == 0 ) test = CERENKOV_PHOTON_K ;
   if(strcmp(name,"E") == 0 ) test = CERENKOV_PHOTON_ENPROP_E ;
   if(strcmp(name,"X") == 0 ) test = CERENKOV_PHOTON_EXPT_X ;
   if(strcmp(name,"G") == 0 ) test = GENERATE_PHOTON_G ;
   if(strcmp(name,"A") == 0 ) test = BOUNDARY_LOOKUP_ALL_A ;
   if(strcmp(name,"L") == 0 ) test = BOUNDARY_LOOKUP_LINE_LS_L ;
   if(strcmp(name,"Y") == 0 ) test = PROP_LOOKUP_Y ;

   if(strcmp(name,"water") == 0    )    test = BOUNDARY_LOOKUP_LINE_WATER_W ;
   if(strcmp(name,"fill_state_0") == 0) test = FILL_STATE_0 ;
   if(strcmp(name,"fill_state_1") == 0) test = FILL_STATE_1 ;
   if(strcmp(name,"propagate_to_boundary") == 0)  test = PROPAGATE_TO_BOUNDARY ;
   if(strcmp(name,"rayleigh_scatter_align") == 0) test = RAYLEIGH_SCATTER_ALIGN ;
   
   bool known =  test != UNKNOWN  ;
   if(!known) LOG(fatal) << " test name " << name << " is unknown " ; 
   assert(known);  
   return test ; 
}

std::string MakeName(const char* prefix, unsigned num, const char* ext)
{
    std::stringstream ss ; 
    ss << prefix ; 
#ifdef FLIP_RANDOM 
    ss << "FLIP_RANDOM_" ; 
#endif
    ss << num << ext ; 
    return ss.str(); 
}


template <typename T>
struct QSimTest
{
    static const char* FOLD ; 

    QSim<T>& qs ; 
    QSimTest(QSim<T>& qs); 
    void main(int argc, char** argv, unsigned test); 

    void rng_sequence(unsigned ni, int ni_tranche_size); 
    void wavelength(char mode, unsigned num_wavelength) ; 

    void scint_photon(unsigned num_photon); 
    void cerenkov_photon(unsigned num_photon, int print_id); 
    void cerenkov_photon_enprop(unsigned num_photon, int print_id); 
    void cerenkov_photon_expt(  unsigned num_photon, int print_id); 

    void generate_photon(); 

    void getStateNames(std::vector<std::string>& names, unsigned num_state) const ; 
    void fill_state(unsigned version); 
    void save_state( const char* subfold, const float* data, unsigned num_state  ); 

    void rayleigh_scatter_align(unsigned num_photon); 
    void propagate_to_boundary(unsigned num_photon); 

    void boundary_lookup_all();
    void boundary_lookup_line(const char* material, T x0, T x1, unsigned nx ); 

    void prop_lookup( int iprop, T x0, T x1, unsigned nx ); 
}; 

template <typename T>
const char* QSimTest<T>::FOLD = SPath::Resolve("$TMP/QSimTest", 2) ;  // 2:dirpath create 

template <typename T>
QSimTest<T>::QSimTest(QSim<T>& qs_)
    :
    qs(qs_)
{
}

template <typename T>
void QSimTest<T>::rng_sequence(unsigned ni, int ni_tranche_size_)
{
    unsigned nj = 16 ; 
    unsigned nk = 16 ; 
    unsigned ni_tranche_size = ni_tranche_size_ > 0 ? ni_tranche_size_ : ni ; 

    qs.rng_sequence(FOLD, ni, nj, nk, ni_tranche_size ); 
}

template <typename T>
void QSimTest<T>::wavelength(char mode, unsigned num_wavelength )
{
    assert( mode == 'S' || mode == 'C' ) ;

    std::vector<T> w(num_wavelength, 0.f) ; 

    std::stringstream ss ; 
    ss << "wavelength" ; ; 
    if( mode == 'S' )
    {
        unsigned hd_factor(~0u) ; 
        qs.scint_wavelength(   w.data(), w.size(), hd_factor );  // hd_factor is an output argument
        assert( hd_factor == 0 || hd_factor == 10 || hd_factor == 20 ); 
        ss << "_scint_hd" << hd_factor ; 
        char scintTexFilterMode = qs.getScintTexFilterMode() ; 
        if(scintTexFilterMode == 'P') ss << "_cudaFilterModePoint" ; 
    }
    else if( mode == 'C' )
    {
        qs.cerenkov_wavelength_rejection_sampled( w.data(), w.size() ); 
        ss << "_cerenkov" ; 
    }

    ss << "_" << num_wavelength << ".npy" ; 
    std::string s = ss.str();
    const char* name = s.c_str(); 

    qs.dump_wavelength( w.data(), w.size() ); 
   
    LOG(info) << " name " << name ; 
    NP::Write( FOLD, name ,  w ); 
}

template <typename T>
void QSimTest<T>::scint_photon(unsigned num_photon)
{
    std::string name = MakeName("photon_", num_photon, ".npy" ); 
    LOG(info) << name ; 

    std::vector<quad4> p(num_photon) ; 
    qs.scint_photon( p.data(), p.size() ); 
    qs.dump_photon(  p.data(), p.size() ); 

    NP::Write( FOLD, name.c_str(),  (float*)p.data(), p.size(), 4, 4  ); 
}

// TODO: elimiate the duplication between these

template <typename T>
void QSimTest<T>::cerenkov_photon(unsigned num_photon, int print_id)
{
    std::string name = MakeName("cerenkov_photon_", num_photon, ".npy" ); 
    LOG(info) << name << " print_id " << print_id ; 
    std::vector<quad4> p(num_photon) ; 
    qs.cerenkov_photon(   p.data(), p.size(), print_id ); // alloc on device, generate and copy back into the vector
    qs.dump_photon(       p.data(), p.size() ); 
    NP::Write( FOLD, name.c_str() ,  (float*)p.data(), p.size(), 4, 4  ); 
}

template <typename T>
void QSimTest<T>::cerenkov_photon_enprop(unsigned num_photon, int print_id)
{
    std::string name = MakeName("cerenkov_photon_enprop_", num_photon, ".npy" ); 
    LOG(info) << name << " print_id " << print_id ; 
    std::vector<quad4> p(num_photon) ; 
    qs.cerenkov_photon_enprop(   p.data(), p.size(), print_id ); 
    qs.dump_photon(              p.data(), p.size() ); 
    NP::Write( FOLD, name.c_str() ,  (float*)p.data(), p.size(), 4, 4  ); 
}

template <typename T>
void QSimTest<T>::cerenkov_photon_expt(unsigned num_photon, int print_id)
{
    std::string name = MakeName("cerenkov_photon_expt_", num_photon, ".npy" ); 
    LOG(info) << name << " print_id " << print_id ; 
    std::vector<quad4> p(num_photon) ; 
    qs.cerenkov_photon_expt(   p.data(), p.size(), print_id ); 
    qs.dump_photon(            p.data(), p.size() ); 
    NP::Write( FOLD, name.c_str() ,  (float*)p.data(), p.size(), 4, 4  ); 
}





template <typename T>
void QSimTest<T>::generate_photon()
{
    LOG(info) << "[" ; 

    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    unsigned x_total = 0 ; 
    for(unsigned i=0 ; i < photon_counts_per_genstep.size() ; i++) x_total += photon_counts_per_genstep[i] ; 

    const NP* gs = SEvent::MakeCountGensteps(photon_counts_per_genstep) ; 


    QEvent* evt = new QEvent  ; 
    evt->setGensteps(gs);
 
    qs.generate_photon(evt);  

    std::vector<quad4> photon ; 
    evt->downloadPhoton(photon); 
    LOG(info) << " downloadPhoton photon.size " << photon.size() ; 

    qs.dump_photon( photon.data(), photon.size(), "f0,f1,i2,i3" ); 

    LOG(info) << "]" ; 
}



/**
QSimTest::fill_state
-----------------------

Doing this for all boundaries with -0.5 and +0.5 for cosTheta will cover 
all states at a particular wavelength 

**/

template <typename T>
void QSimTest<T>::fill_state(unsigned version)
{
    LOG(info) << "[" ; 

    unsigned num_state = qs.bnd->getNumBoundary() ; 

    if( version == 0 )
    {
        std::vector<quad6> s(num_state) ; 
        qs.fill_state_0( s.data(), s.size() ); 
        save_state("fill_state_0", (float*)s.data(), num_state );
    }
    else if( version == 1 )
    {
        std::vector<qstate> s(num_state) ; 
        qs.fill_state_1( s.data(), s.size() ); 
        save_state("fill_state_1", (float*)s.data(), num_state );
    }
    LOG(info) << "]" ; 
}


template <typename T>
void QSimTest<T>::save_state( const char* subfold, const float* data, unsigned num_state  )
{
    std::vector<std::string> names ; 
    getStateNames(names, num_state); 

    int create_dirs = 1 ; // 1:filepath 
    const char* path = SPath::Resolve(FOLD, subfold, "state.npy", create_dirs ); 

    NP* a = NP::Make<float>( num_state, 6, 4 ); // (6,4) item dimension corresponds to the 6 quads of quad6 
    a->read( data ); 
    a->set_names(names); 
    a->save(path); 
}


template <typename T>
void QSimTest<T>::getStateNames(std::vector<std::string>& names, unsigned num_state) const 
{
    unsigned* idx = new unsigned[num_state] ; 
    for(unsigned i=0 ; i < num_state ; i++) idx[i] = i ; 
    qs.bnd->getBoundarySpec(names, idx, num_state ); 
    delete [] idx ; 
}

template <typename T>
void QSimTest<T>::rayleigh_scatter_align(unsigned num_photon)
{
    LOG(info); 
    std::vector<quad4> p(num_photon) ; 
    qs.rayleigh_scatter_align( p.data(), p.size() ); 
    int create_dirs = 1 ; // 1:filepath 
    const char* path = SPath::Resolve(FOLD, "rayleigh_scatter_align", "p.npy", create_dirs ); 
    NP::Write( path, (float*)p.data(), p.size(), 4, 4  ); 
}

 
template <typename T>
void QSimTest<T>::propagate_to_boundary(unsigned num_photon)
{
    LOG(info); 
    std::vector<quad4> p(num_photon) ; 
    qs.propagate_to_boundary( p.data(), p.size() ); 
    int create_dirs = 1 ; // 1:filepath 
    const char* path = SPath::Resolve(FOLD, "propagate_to_boundary", "p.npy", create_dirs ); 
    NP::Write( path, (float*)p.data(), p.size(), 4, 4  ); 
}


/**
QSimTest::boundary_lookup_all
-------------------------------

Does lookups at every texel of the 2d float4 boundary texture 

**/

template <typename T>
void QSimTest<T>::boundary_lookup_all()
{
    LOG(info); 
    unsigned width = qs.getBoundaryTexWidth(); 
    unsigned height = qs.getBoundaryTexHeight(); 
    const NP* src = qs.getBoundaryTexSrc(); 
    unsigned num_lookup = width*height ; 

    std::vector<quad> lookup(num_lookup); 
    qs.boundary_lookup_all( lookup.data(), width, height ); 

    NP::Write( FOLD, "boundary_lookup_all.npy" ,  (float*)lookup.data(), height, width, 4 ); 
    src->save( FOLD, "boundary_lookup_all_src.npy" ); 
}



/**
QSimTest::boundary_lookup_line
-------------------------------

hmm need templated quad for this to work with T=double 
as its relying on 4*T = quad 

Actually no, if just narrow to float/quad at output  


**/
template <typename T>
void QSimTest<T>::boundary_lookup_line(const char* material, T x0 , T x1, unsigned nx )
{
    LOG(info); 

    unsigned line = qs.bnd->getMaterialLine(material); 
    if( line == ~0u )
    {
        LOG(fatal) << " material not in boundary tex " << material ; 
        assert(0); 
    }

    LOG(info) << " material " << material << " line " << line ; 
    unsigned k = 0 ;    // 0 or 1 picking the property float4 group to collect 


    NP* x = NP::Linspace<T>(x0,x1,nx); 
    T* xx = x->values<T>(); 

    std::vector<quad> lookup(nx) ; 
    qs.boundary_lookup_line( lookup.data(), xx, nx, line, k ); 

    NP::Write( FOLD, "boundary_lookup_line_props.npy" ,    (float*)lookup.data(), nx, 4  ); 
    NP::Write( FOLD, "boundary_lookup_line_wavelength.npy" ,  xx, nx ); 
}

template<typename T>
void QSimTest<T>::prop_lookup( int iprop, T x0, T x1, unsigned nx )
{
    unsigned tot_prop = qs.prop->ni ; 
    const NP* pp = qs.prop->a ; 

    std::vector<unsigned> pids ; 
    if( iprop == -1 )
    { 
        for(unsigned i=0 ; i < tot_prop ; i++ ) pids.push_back(i);
    }
    else
    {
        pids.push_back(iprop); 
    } 

    unsigned num_prop = pids.size() ; 

    LOG(info) 
        << " tot_prop " << tot_prop
        << " iprop " << iprop
        << " pids.size " << pids.size()
        << " num_prop " << num_prop 
        << " pp " << pp->desc()
        ; 


    NP* yy = NP::Make<T>(num_prop, nx) ; 
    NP* x = NP::Linspace<T>(x0,x1,nx); 

    qs.prop_lookup_onebyone( yy->values<T>(), x->cvalues<T>(), nx, pids ) ;

    const char* reldir = sizeof(T) == 8 ? "double" : "float" ; 

    pp->save(FOLD, reldir, "prop_lookup_pp.npy" ); 
    x->save(FOLD, reldir, "prop_lookup_x.npy" ); 
    yy->save(FOLD, reldir, "prop_lookup_yy.npy" ); 
}



template<typename T>
void QSimTest<T>::main(int argc, char** argv, unsigned test )
{
    unsigned num_default = SSys::getenvunsigned("NUM", 1000000u )  ;   
    unsigned num = argc > 1 ? std::atoi(argv[1]) : num_default ; 
    int ni_tranche_size = SSys::getenvint("NI_TRANCHE_SIZE", 100000 ); // default 100k usable with any GPU 
    int print_id = SSys::getenvint("PINDEX", -1 ); 

    LOG(info) 
        << " num_default " << num_default 
        << " num " << num 
        << " test " << test
        << " ni_tranche_size " << ni_tranche_size
        << " print_id " << print_id
        ; 

    T x0 = 80. ; 
    T x1 = 800. ; 
    unsigned nx = 721u ; 

    switch(test)
    {
        case RNG_SEQUENCE_F:                rng_sequence(num, ni_tranche_size)         ; break ; 
        case WAVELENGTH_S  :                wavelength('S', num)                       ; break ; 
        case WAVELENGTH_C  :                wavelength('C', num)                       ; break ; 
        case SCINT_PHOTON_P:                scint_photon(num);                         ; break ; 
        case CERENKOV_PHOTON_K:             cerenkov_photon(num, print_id);            ; break ; 
        case CERENKOV_PHOTON_ENPROP_E:      cerenkov_photon_enprop(num, print_id);     ; break ; 
        case CERENKOV_PHOTON_EXPT_X :       cerenkov_photon_expt(  num, print_id);     ; break ; 
        case GENERATE_PHOTON_G:             generate_photon();                         ; break ; 
        case BOUNDARY_LOOKUP_ALL_A:         boundary_lookup_all()                      ; break ;  
        case BOUNDARY_LOOKUP_LINE_WATER_W:  boundary_lookup_line("Water", x0, x1, nx)  ; break ;  
        case BOUNDARY_LOOKUP_LINE_LS_L:     boundary_lookup_line("LS",    x0, x1, nx)  ; break ;  
        case PROP_LOOKUP_Y:                 prop_lookup(-1, -1.f,16.f,1701)            ; break ;  
        case FILL_STATE_0:                  fill_state(0)                              ; break ;  
        case FILL_STATE_1:                  fill_state(1)                              ; break ;  
        case PROPAGATE_TO_BOUNDARY:         propagate_to_boundary(8)                   ; break ;  
        case RAYLEIGH_SCATTER_ALIGN:        rayleigh_scatter_align(8)                  ; break ;   
        default :                           LOG(fatal) << "unimplemented" << std::endl ; break ; 
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

#ifdef OLD
    Opticks ok(argc, argv); 
    ok.configure(); 
    const char* idpath = ok.getIdPath(); 
    const char* cfbase = ok.getFoundryBase("CFBASE") ; 
#else
    const char* idpath = SOpticksResource::IDPath(true);  
    const char* cfbase = SOpticksResource::CFBase(); 
#endif

    int create_dirs = 0 ; 
    const char* rindexpath = SPath::Resolve(idpath, "GScintillatorLib/LS_ori/RINDEX.npy", create_dirs );  
    // HMM: this is just for one material ... Cerenkov needs this for all 

    // TODO: need better icdf naming now that can do both scint and ck with icdf, see CSG_GGeo_Convert::convertScintillatorLib 
    NP* icdf    = NP::Load(cfbase, "CSGFoundry", "icdf.npy");  
    NP* bnd     = NP::Load(cfbase, "CSGFoundry", "bnd.npy"); 
    NP* optical = NP::Load(cfbase, "CSGFoundry", "optical.npy"); 

    if(icdf == nullptr || bnd == nullptr)
    {
        LOG(fatal) 
            << " MISSING QSim CSGFoundry input arrays "
            << " cfbase " << cfbase 
            << " icdf " << icdf 
            << " bnd " << bnd 
            << " (recreate these with : \"c ; om ; cg ; om ; ./run.sh \" ) "
            ;
        return 1 ; 
    }

    const char* default_testname = "G" ; 
    const char* testname = SSys::getenvvar("TEST", default_testname); 
    int test = TestType(testname); 
    char type = SSys::getenvchar("TYPE", 'F'); 

    if( test == CERENKOV_PHOTON_EXPT_X ) type = 'D' ;   // forced double 

    if( type == 'F')
    { 
        LOG(error) << "[ QSim<float>::UploadComponents" ; 
        QSim<float>::UploadComponents(icdf, bnd, optical, rindexpath ); 
        LOG(error) << "] QSim<float>::UploadComponents" ; 
        QSim<float> qs ; 
        QSimTest<float> qst(qs) ; 
        qst.main( argc, argv, test ); 
    }
    else if( type == 'D' )
    {
        QSim<double>::UploadComponents(icdf, bnd, optical, rindexpath ); 
        QSim<double> qs ; 
        QSimTest<double> qst(qs) ; 
        qst.main( argc, argv, test ); 
    }
    cudaDeviceSynchronize();
    return 0 ; 
}
