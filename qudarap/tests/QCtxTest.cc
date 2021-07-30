#include <sstream>

#include "Opticks.hh"
#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"
#include "NPY.hpp"
#include "GGeo.hh"
#include "GBndLib.hh"
#include "GScintillatorLib.hh"

#include "QRng.hh"
#include "QBnd.hh"
#include "QProp.hh"
#include "QCtx.hh"
#include "scuda.h"

#include "OPTICKS_LOG.hh"


template <typename T>
struct QCtxTest
{
    static const char* FOLD ; 
    static std::string MakeName(const char* prefix, unsigned num, const char* ext);

    QCtx<T>& qc ; 
    QCtxTest(QCtx<T>& qc); 
    void main(int argc, char** argv, char test); 

    void rng_sequence_0(unsigned num_rng); 
    void rng_sequence(unsigned ni, int ni_tranche_size); 
    void wavelength(char mode, unsigned num_wavelength) ; 

    void scint_photon(unsigned num_photon); 
    void cerenkov_photon(unsigned num_photon, int print_id); 
    void cerenkov_photon_enprop(unsigned num_photon, int print_id); 
    void cerenkov_photon_expt(  unsigned num_photon, int print_id); 

    void boundary_lookup_all();
    void boundary_lookup_line(const char* material, T x0, T x1, unsigned nx ); 

    void prop_lookup( int iprop, T x0, T x1, unsigned nx ); 

}; 

template <typename T>
const char* QCtxTest<T>::FOLD = "/tmp/QCtxTest" ; 

template <typename T>
QCtxTest<T>::QCtxTest(QCtx<T>& qc_)
    :
    qc(qc_) 
{
}


template <typename T>
void QCtxTest<T>::rng_sequence_0(unsigned num_rng )
{
    std::vector<T> rs(num_rng) ; 
    qc.rng_sequence_0( rs.data(), rs.size() ); 

    std::string name = MakeName("rng_sequence_0_", num_rng, ".npy" ); 
    NP::Write( FOLD, name.c_str() ,  rs ); 
}

template <typename T>
void QCtxTest<T>::rng_sequence(unsigned ni, int ni_tranche_size_)
{
    unsigned nj = 16 ; 
    unsigned nk = 16 ; 
    unsigned ni_tranche_size = ni_tranche_size_ > 0 ? ni_tranche_size_ : ni ; 

    qc.rng_sequence(FOLD, ni, nj, nk, ni_tranche_size ); 
}


template <typename T>
void QCtxTest<T>::wavelength(char mode, unsigned num_wavelength )
{
    assert( mode == 'S' || mode == 'C' ) ;

    std::vector<T> w(num_wavelength, 0.f) ; 

    std::stringstream ss ; 
    ss << "wavelength" ; ; 
    if( mode == 'S' )
    {
        unsigned hd_factor(~0u) ; 
        qc.scint_wavelength(   w.data(), w.size(), hd_factor );  // hd_factor is an output argument
        assert( hd_factor == 0 || hd_factor == 10 || hd_factor == 20 ); 
        ss << "_scint_hd" << hd_factor ; 
        char scintTexFilterMode = qc.getScintTexFilterMode() ; 
        if(scintTexFilterMode == 'P') ss << "_cudaFilterModePoint" ; 
    }
    else if( mode == 'C' )
    {
        qc.cerenkov_wavelength( w.data(), w.size() ); 
        ss << "_cerenkov" ; 
    }

    ss << "_" << num_wavelength << ".npy" ; 
    std::string s = ss.str();
    const char* name = s.c_str(); 

    qc.dump_wavelength( w.data(), w.size() ); 
   
    LOG(info) << " name " << name ; 
    NP::Write( FOLD, name ,  w ); 
}

template <typename T>
void QCtxTest<T>::scint_photon(unsigned num_photon)
{
    std::string name = MakeName("photon_", num_photon, ".npy" ); 
    LOG(info) << name ; 

    std::vector<quad4> p(num_photon) ; 
    qc.scint_photon( p.data(), p.size() ); 
    qc.dump_photon(  p.data(), p.size() ); 

    NP::Write( FOLD, name.c_str(),  (float*)p.data(), p.size(), 4, 4  ); 
}


template <typename T>
std::string QCtxTest<T>::MakeName(const char* prefix, unsigned num, const char* ext)
{
    std::stringstream ss ; 
    ss << prefix ; 
#ifdef FLIP_RANDOM 
    ss << "FLIP_RANDOM_" ; 
#endif
    ss << num << ext ; 
    return ss.str(); 
}



// TODO: elimiate the duplication between these

template <typename T>
void QCtxTest<T>::cerenkov_photon(unsigned num_photon, int print_id)
{
    std::string name = MakeName("cerenkov_photon_", num_photon, ".npy" ); 
    LOG(info) << name << " print_id " << print_id ; 
    std::vector<quad4> p(num_photon) ; 
    qc.cerenkov_photon(   p.data(), p.size(), print_id ); 
    qc.dump_photon(       p.data(), p.size() ); 
    NP::Write( FOLD, name.c_str() ,  (float*)p.data(), p.size(), 4, 4  ); 
}

template <typename T>
void QCtxTest<T>::cerenkov_photon_enprop(unsigned num_photon, int print_id)
{
    std::string name = MakeName("cerenkov_photon_enprop_", num_photon, ".npy" ); 
    LOG(info) << name << " print_id " << print_id ; 
    std::vector<quad4> p(num_photon) ; 
    qc.cerenkov_photon_enprop(   p.data(), p.size(), print_id ); 
    qc.dump_photon(              p.data(), p.size() ); 
    NP::Write( FOLD, name.c_str() ,  (float*)p.data(), p.size(), 4, 4  ); 
}

template <typename T>
void QCtxTest<T>::cerenkov_photon_expt(unsigned num_photon, int print_id)
{
    std::string name = MakeName("cerenkov_photon_expt_", num_photon, ".npy" ); 
    LOG(info) << name << " print_id " << print_id ; 
    std::vector<quad4> p(num_photon) ; 
    qc.cerenkov_photon_expt(   p.data(), p.size(), print_id ); 
    qc.dump_photon(            p.data(), p.size() ); 
    NP::Write( FOLD, name.c_str() ,  (float*)p.data(), p.size(), 4, 4  ); 
}







/**
QCtxTest::boundary_lookup_all
-------------------------------

Does lookups at every texel of the 2d float4 boundary texture 

**/

template <typename T>
void QCtxTest<T>::boundary_lookup_all()
{
    LOG(info); 
    unsigned width = qc.getBoundaryTexWidth(); 
    unsigned height = qc.getBoundaryTexHeight(); 
    const NPY<float>* src = qc.getBoundaryTexSrc(); 
    unsigned num_lookup = width*height ; 

    std::vector<quad> lookup(num_lookup); 
    qc.boundary_lookup_all( lookup.data(), width, height ); 

    NP::Write( FOLD, "boundary_lookup_all.npy" ,  (float*)lookup.data(), height, width, 4 ); 
    src->save( FOLD, "boundary_lookup_all_src.npy" ); 
}



/**
QCtxTest::boundary_lookup_line
-------------------------------

hmm need templated quad for this to work with T=double 
as its relying on 4*T = quad 

Actually no, if just narrow to float/quad at output  


**/
template <typename T>
void QCtxTest<T>::boundary_lookup_line(const char* material, T x0 , T x1, unsigned nx )
{
    LOG(info); 

    unsigned line = qc.bnd->getMaterialLine(material); 
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
    qc.boundary_lookup_line( lookup.data(), xx, nx, line, k ); 

    NP::Write( FOLD, "boundary_lookup_line_props.npy" ,    (float*)lookup.data(), nx, 4  ); 
    NP::Write( FOLD, "boundary_lookup_line_wavelength.npy" ,  xx, nx ); 
}




template<typename T>
void QCtxTest<T>::prop_lookup( int iprop, T x0, T x1, unsigned nx )
{
    unsigned tot_prop = qc.prop->ni ; 
    const NP* pp = qc.prop->a ; 

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

    qc.prop_lookup_onebyone( yy->values<T>(), x->cvalues<T>(), nx, pids ) ;

    const char* reldir = sizeof(T) == 8 ? "double" : "float" ; 

    pp->save(FOLD, reldir, "prop_lookup_pp.npy" ); 
    x->save(FOLD, reldir, "prop_lookup_x.npy" ); 
    yy->save(FOLD, reldir, "prop_lookup_yy.npy" ); 
}


template<typename T>
void QCtxTest<T>::main(int argc, char** argv, char test )
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
        case '0': rng_sequence_0(num)                        ; break ; 
        case 'F': rng_sequence(num, ni_tranche_size)         ; break ; 
        case 'S': wavelength('S', num)                       ; break ; 
        case 'C': wavelength('C', num)                       ; break ; 
        case 'P': scint_photon(num);                         ; break ; 
        case 'K': cerenkov_photon(num, print_id);            ; break ; 
        case 'E': cerenkov_photon_enprop(num, print_id);     ; break ; 
        case 'X': cerenkov_photon_expt(  num, print_id);     ; break ; 
        case 'A': boundary_lookup_all()                      ; break ;  
        case 'W': boundary_lookup_line("Water", x0, x1, nx)  ; break ;  
        case 'L': boundary_lookup_line("LS",    x0, x1, nx)  ; break ;  
        case 'Y': prop_lookup(-1, -1.f,16.f,1701)            ; break ;  
        default : std::cout << "test unimplemented" << std::endl ; break ; 
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 
    GGeo* gg = GGeo::Load(&ok); 

    char test = SSys::getenvchar("TEST", 'X'); 
    char type = SSys::getenvchar("TYPE", 'F'); 
    if( test == 'X') type = 'D' ;   // forced double 

    if( type == 'F')
    { 
        QCtx<float>::Init(gg); 
        QCtx<float> qc ; 
        QCtxTest<float> qtc(qc) ; 
        qtc.main( argc, argv, test ); 
    }
    else if( type == 'D' )
    {
        QCtx<double>::Init(gg); 
        QCtx<double> qc ; 
        QCtxTest<double> qtc(qc) ; 
        qtc.main( argc, argv, test ); 
    }

    return 0 ; 
}
