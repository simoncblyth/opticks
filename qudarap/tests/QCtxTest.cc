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
    QCtx qc ; 
    QCtxTest(QCtx& qc_); 

    void wavelength(char mode) ; 
    void photon(); 
    void boundary_lookup_all();
    void boundary_lookup_line();
}; 

const char* QCtxTest::FOLD = "/tmp/QCtxTest" ; 

QCtxTest::QCtxTest(QCtx& qc_)
    :
    qc(qc_)
{
}

void QCtxTest::wavelength(char mode)
{
    assert( mode == 'S' || mode == 'C' ) ;
    unsigned num_wavelength = 1000000 ; 

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
        qc.generate_cerenkov(   w.data(), w.size() ); 
        ss << "_cerenkov" ; 
    }
    ss << ".npy" ; 
    std::string s = ss.str();
    const char* name = s.c_str(); 

    qc.dump(             w.data(), w.size() ); 
   
    LOG(info) << " name " << name ; 
    NP::Write( FOLD, name ,  w ); 
}

void QCtxTest::photon()
{
    LOG(info); 
    unsigned num_photon = 100 ; 
    std::vector<quad4> p ; 
    p.resize(num_photon); 

    qc.generate(   p.data(), p.size() ); 
    qc.dump(       p.data(), p.size() ); 
    NP::Write( FOLD, "photon.npy" ,  (float*)p.data(), p.size(), 4, 4  ); 
}

/**
test_boundary_lookup
----------------------

Does lookups at every point of the 2d float4 boundary texture 

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


void QCtxTest::boundary_lookup_line()
{
    LOG(info); 

    unsigned num_lookup = 100 ; 
    std::vector<quad> lookup(num_lookup); 

    const char* spec = SSys::getenvvar("QCTX_SPEC", "Acrylic///LS" ); 
    unsigned imat = 3 ;  // picking LS 
    unsigned line = qc.bnd->getBoundaryLine(spec, imat); 
    unsigned k = 0 ; 

    float nm0 = 300.f ; 
    float nm1 = 600.f ; 
    std::vector<float> domain(num_lookup); 
    for(unsigned i=0 ; i < num_lookup ; i++) domain[i] = nm0 + (nm1 - nm0)*float(i)/float(num_lookup) ; 

    qc.boundary_lookup_line( lookup.data(), domain.data(), num_lookup, line, k ); 

    NP::Write( FOLD, "boundary_lookup_line_props.npy" ,       (float*)lookup.data(), num_lookup, 4  ); 
    NP::Write( FOLD, "boundary_lookup_line_wavelength.npy" ,          domain.data(), num_lookup ); 
}


int main(int argc, char** argv)
{
    char test = argc > 1 ? argv[1][0] : 'L' ; 
    OPTICKS_LOG(argc, argv); 
    LOG(info) << " test " << test ; 

    Opticks ok(argc, argv); 
    ok.configure(); 
    GGeo* gg = GGeo::Load(&ok); 

    QCtx::Init(gg); 
    QCtx qc ;  

    QCtxTest qtc(qc); 
    switch(test)
    {
        case 'S': qtc.wavelength('S')        ; break ; 
        case 'C': qtc.wavelength('C')        ; break ; 
        case 'P': qtc.photon();              ; break ; 
        case 'A': qtc.boundary_lookup_all()  ; break ;  
        case 'L': qtc.boundary_lookup_line() ; break ;  
    }
    return 0 ; 
}
