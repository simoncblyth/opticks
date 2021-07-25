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


struct QCtxTest
{
    static const char* FOLD ; 
    static std::string MakeName(const char* prefix, unsigned num, const char* ext);

    QCtx qc ; 
    QCtxTest(QCtx& qc_); 

    void rng_sequence_0(unsigned num_rng); 
    void rng_sequence_f(unsigned ni, int ni_tranche_size); 
    void wavelength(char mode, unsigned num_wavelength) ; 

    void scint_photon(unsigned num_photon); 
    void cerenkov_photon(unsigned num_photon, int print_id); 
    void cerenkov_photon_enprop(unsigned num_photon, int print_id); 

    void boundary_lookup_all();
    void boundary_lookup_line(const char* material, double nm0=80., double nm1=800., double nm_step=1. ); 
    void prop_lookup( int iprop=-1, float x0=0.f, float x1=10.f, unsigned nx=101u  ); 

}; 

const char* QCtxTest::FOLD = "/tmp/QCtxTest" ; 

QCtxTest::QCtxTest(QCtx& qc_)
    :
    qc(qc_)
{
}

void QCtxTest::rng_sequence_0(unsigned num_rng )
{
    std::vector<float> rs ; 
    rs.resize(num_rng, 0.f); 
    qc.rng_sequence_0( rs.data(), rs.size() ); 

    std::string name = MakeName("rng_sequence_0_", num_rng, ".npy" ); 
    NP::Write( FOLD, name.c_str() ,  rs ); 
}

void QCtxTest::rng_sequence_f(unsigned ni, int ni_tranche_size_)
{
    unsigned nj = 16 ; 
    unsigned nk = 16 ; 
    unsigned ni_tranche_size = ni_tranche_size_ > 0 ? ni_tranche_size_ : ni ; 

    qc.rng_sequence<float>(FOLD, ni, nj, nk, ni_tranche_size ); 
}


void QCtxTest::wavelength(char mode, unsigned num_wavelength )
{
    assert( mode == 'S' || mode == 'C' ) ;

    std::vector<float> w(num_wavelength, 0.f) ; 

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

void QCtxTest::scint_photon(unsigned num_photon)
{
    std::string name = MakeName("photon_", num_photon, ".npy" ); 
    LOG(info) << name ; 

    std::vector<quad4> p(num_photon) ; 
    qc.scint_photon( p.data(), p.size() ); 
    qc.dump_photon(  p.data(), p.size() ); 

    NP::Write( FOLD, name.c_str(),  (float*)p.data(), p.size(), 4, 4  ); 
}


std::string QCtxTest::MakeName(const char* prefix, unsigned num, const char* ext)
{
    std::stringstream ss ; 
    ss << prefix << num << ext ; 
    return ss.str(); 
}


void QCtxTest::cerenkov_photon(unsigned num_photon, int print_id)
{
#ifdef FLIP_RANDOM 
    std::string name = MakeName("cerenkov_photon_FLIP_RANDOM_", num_photon, ".npy" ); 
#else
    std::string name = MakeName("cerenkov_photon_", num_photon, ".npy" ); 
#endif
    LOG(info) << name << " print_id " << print_id ; 
    std::vector<quad4> p(num_photon) ; 
    qc.cerenkov_photon(   p.data(), p.size(), print_id ); 
    qc.dump_photon(       p.data(), p.size() ); 
    NP::Write( FOLD, name.c_str() ,  (float*)p.data(), p.size(), 4, 4  ); 
}


void QCtxTest::cerenkov_photon_enprop(unsigned num_photon, int print_id)
{
#ifdef FLIP_RANDOM 
    std::string name = MakeName("cerenkov_photon_enprop_FLIP_RANDOM_", num_photon, ".npy" ); 
#else
    std::string name = MakeName("cerenkov_photon_enprop_", num_photon, ".npy" ); 
#endif
    LOG(info) << name << " print_id " << print_id ; 
    std::vector<quad4> p(num_photon) ; 
    qc.cerenkov_photon_enprop(   p.data(), p.size(), print_id ); 
    qc.dump_photon(              p.data(), p.size() ); 
    NP::Write( FOLD, name.c_str() ,  (float*)p.data(), p.size(), 4, 4  ); 
}



/**
QCtxTest::boundary_lookup_all
-------------------------------

Does lookups at every texel of the 2d float4 boundary texture 

**/

void QCtxTest::boundary_lookup_all()
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


**/
void QCtxTest::boundary_lookup_line(const char* material, double nm0 , double nm1, double nm_step )
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

    std::vector<double> domain ; 
    for(double nm=nm0 ; nm < nm1 ; nm += nm_step ) domain.push_back(nm) ; 
    unsigned num_lookup = domain.size() ; 
    LOG(info) << " nm0 " << nm0 << " nm1 " << nm1 << " nm_step " << nm_step << " num_lookup " << num_lookup ; 

    std::vector<float> fdomain(domain.size()); 
    for(unsigned i=0 ; i < domain.size() ; i++ ) fdomain[i] = float(domain[i]) ;  

    std::vector<quad> lookup(num_lookup) ; 

    qc.boundary_lookup_line( lookup.data(), fdomain.data(), num_lookup, line, k ); 


    NP::Write( FOLD, "boundary_lookup_line_props.npy" ,       (float*)lookup.data(), num_lookup, 4  ); 
    NP::Write( FOLD, "boundary_lookup_line_wavelength.npy" ,          domain.data(), num_lookup ); 
}


void QCtxTest::prop_lookup( int iprop, float x0, float x1, unsigned nx  )
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


    NP* yy = NP::Make<float>(num_prop, nx) ; 
    NP* x = NP::Linspace<float>(x0,x1,nx); 

    //qc.prop_lookup( yy->values<float>(), x->cvalues<float>(), nx, pids ) ;
    qc.prop_lookup_onebyone( yy->values<float>(), x->cvalues<float>(), nx, pids ) ;

    pp->save(FOLD, "prop_lookup_pp.npy" ); 
    x->save(FOLD, "prop_lookup_x.npy" ); 
    yy->save(FOLD, "prop_lookup_yy.npy" ); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    unsigned num_default = SSys::getenvunsigned("NUM", 1000000u )  ;   
    unsigned num = argc > 1 ? std::atoi(argv[1]) : num_default ; 
    char test = SSys::getenvchar("TEST", 'E'); 
    int ni_tranche_size = SSys::getenvint("NI_TRANCHE_SIZE", 100000 ); // default 100k usable with any GPU 
    int print_id = SSys::getenvint("PINDEX", -1 ); 

    LOG(info) 
        << " num_default " << num_default 
        << " num " << num 
        << " test " << test
        << " ni_tranche_size " << ni_tranche_size
        << " print_id " << print_id
        ; 

    Opticks ok(argc, argv); 
    ok.configure(); 
    GGeo* gg = GGeo::Load(&ok); 

    QCtx::Init(gg); 
    QCtx qc ;  
    QCtxTest qtc(qc); 

    switch(test)
    {
        case '0': qtc.rng_sequence_0(num)                        ; break ; 
        case 'F': qtc.rng_sequence_f(num, ni_tranche_size)       ; break ; 
        case 'S': qtc.wavelength('S', num)                       ; break ; 
        case 'C': qtc.wavelength('C', num)                       ; break ; 
        case 'P': qtc.scint_photon(num);                         ; break ; 
        case 'K': qtc.cerenkov_photon(num, print_id);            ; break ; 
        case 'E': qtc.cerenkov_photon_enprop(num, print_id);     ; break ; 
        case 'A': qtc.boundary_lookup_all()                      ; break ;  
        case 'W': qtc.boundary_lookup_line("Water")              ; break ;  
        case 'L': qtc.boundary_lookup_line("LS",80., 800., 0.1)  ; break ;  
        case 'Y': qtc.prop_lookup(-1, -1.f,16.f,1701)            ; break ;  
        default : std::cout << "test unimplemented" << std::endl ; break ; 
    }
    return 0 ; 
}
