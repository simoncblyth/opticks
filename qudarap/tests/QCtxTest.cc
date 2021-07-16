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
#include "QCtx.hh"
#include "scuda.h"

#include "OPTICKS_LOG.hh"


struct QCtxTest
{
    static const char* FOLD ; 
    static std::string MakeName(const char* prefix, unsigned num, const char* ext);

    QCtx qc ; 
    QCtxTest(QCtx& qc_); 

    void rng_sequence(unsigned num_rng); 
    void wavelength(char mode, unsigned num_wavelength) ; 
    void photon(unsigned num_photon); 
    void cerenkov_photon(unsigned num_photon); 
    void boundary_lookup_all();
    void boundary_lookup_line(const char* material, double nm0=80., double nm1=800., double nm_step=1. ); 
}; 

const char* QCtxTest::FOLD = "/tmp/QCtxTest" ; 

QCtxTest::QCtxTest(QCtx& qc_)
    :
    qc(qc_)
{
}

void QCtxTest::rng_sequence(unsigned num_rng )
{
    std::vector<float> rs ; 
    rs.resize(num_rng, 0.f); 
    qc.rng_sequence( rs.data(), rs.size() ); 

    std::string name = MakeName("rng_sequence_", num_rng, ".npy" ); 
    NP::Write( FOLD, name.c_str() ,  rs ); 
}



void QCtxTest::wavelength(char mode, unsigned num_wavelength )
{
    assert( mode == 'S' || mode == 'C' ) ;

    std::vector<float> w ; 
    w.resize(num_wavelength, 0.f); 

    std::stringstream ss ; 
    ss << "wavelength" ; ; 
    if( mode == 'S' )
    {
        unsigned hd_factor(~0u) ; 
        qc.generate_scint(   w.data(), w.size(), hd_factor );  // hd_factor is an output argument
        assert( hd_factor == 0 || hd_factor == 10 || hd_factor == 20 ); 
        ss << "_scint_hd" << hd_factor ; 
        char scintTexFilterMode = qc.getScintTexFilterMode() ; 
        if(scintTexFilterMode == 'P') ss << "_cudaFilterModePoint" ; 
    }
    else if( mode == 'C' )
    {
        qc.generate_cerenkov( w.data(), w.size() ); 
        ss << "_cerenkov" ; 
    }

    ss << "_" << num_wavelength << ".npy" ; 
    std::string s = ss.str();
    const char* name = s.c_str(); 

    qc.dump(             w.data(), w.size() ); 
   
    LOG(info) << " name " << name ; 
    NP::Write( FOLD, name ,  w ); 
}

void QCtxTest::photon(unsigned num_photon)
{
    LOG(info); 
    std::vector<quad4> p ; 
    p.resize(num_photon); 

    qc.generate(   p.data(), p.size() ); 
    qc.dump(       p.data(), p.size() ); 

    std::string name = MakeName("photon_", num_photon, ".npy" ); 
    NP::Write( FOLD, name.c_str() ,  (float*)p.data(), p.size(), 4, 4  ); 
}


std::string QCtxTest::MakeName(const char* prefix, unsigned num, const char* ext)
{
    std::stringstream ss ; 
    ss << prefix << num << ext ; 
    return ss.str(); 
}


void QCtxTest::cerenkov_photon(unsigned num_photon)
{
    LOG(info); 
    std::vector<quad4> p ; 
    p.resize(num_photon); 

    qc.generate_cerenkov_photon(   p.data(), p.size() ); 
    qc.dump(       p.data(), p.size() ); 

  
    std::string name = MakeName("cerenkov_photon_", num_photon, ".npy" ); 
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


int main(int argc, char** argv)
{
    //unsigned num_default = 2820932 ; 
    unsigned num_default = 10000 ; 
    //unsigned num_default = 3000000 ; 

    unsigned num = argc > 1 ? std::atoi(argv[1]) : num_default ; 
    char test = argc > 2 ? argv[2][0] : 'K' ; 
    
    OPTICKS_LOG(argc, argv); 
    LOG(info) << " num " << num << " test " << test ; 

    Opticks ok(argc, argv); 
    ok.configure(); 
    GGeo* gg = GGeo::Load(&ok); 

    QCtx::Init(gg); 
    QCtx qc ;  

    QCtxTest qtc(qc); 
    switch(test)
    {
        case 'R': qtc.rng_sequence(num)             ; break ; 
        case 'S': qtc.wavelength('S', num)          ; break ; 
        case 'C': qtc.wavelength('C', num)          ; break ; 
        case 'P': qtc.photon(num);                  ; break ; 
        case 'K': qtc.cerenkov_photon(num);         ; break ; 
        case 'A': qtc.boundary_lookup_all()         ; break ;  
        case 'W': qtc.boundary_lookup_line("Water") ; break ;  
        case 'L': qtc.boundary_lookup_line("LS",80., 800., 0.1)    ; break ;  
    }
    return 0 ; 
}
