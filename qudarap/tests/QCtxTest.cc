#include <sstream>

#include "Opticks.hh"
#include "SPath.hh"
#include "NP.hh"
#include "NPY.hpp"
#include "GGeo.hh"
#include "GBndLib.hh"
#include "GScintillatorLib.hh"

#include "QRng.hh"
#include "QCtx.hh"
#include "scuda.h"

#include "OPTICKS_LOG.hh"


struct QCtxTest
{
    static const char* FOLD ; 
    QCtx qc ; 

    QCtxTest(QCtx& qc_) : qc(qc_) {}

    void test_wavelength(char mode) ; 
    void test_photon(); 
    void test_boundary_lookup();
}; 


const char* QCtxTest::FOLD = "/tmp/QCtxTest" ; 


void QCtxTest::test_wavelength(char mode)
{
    assert( mode == 'S' || mode == 'C' ) ;
    unsigned num_wavelength = 1000000 ; 

    std::vector<float> wavelength ; 
    wavelength.resize(num_wavelength, 0.f); 

    std::stringstream ss ; 
    ss << "wavelength" ; ; 
    if( mode == 'S' )
    {
        unsigned hd_factor(~0u) ; 
        qc.generate_scint(   wavelength.data(), wavelength.size(), hd_factor );  // hd_factor is an output argument
        assert( hd_factor == 0 || hd_factor == 10 || hd_factor == 20 ); 
        ss << "_scint_hd" << hd_factor ; 
        char scintTexFilterMode = qc.getScintTexFilterMode() ; 
        if(scintTexFilterMode == 'P') ss << "_cudaFilterModePoint" ; 
    }
    else if( mode == 'C' )
    {
        qc.generate_cerenkov(   wavelength.data(), wavelength.size() ); 
        ss << "_cerenkov" ; 
    }
    ss << ".npy" ; 
    std::string s = ss.str();
    const char* name = s.c_str(); 

    qc.dump(             wavelength.data(), wavelength.size() ); 
   
    LOG(info) << " name " << name ; 
    NP::Write( FOLD, name ,  wavelength ); 
}

void QCtxTest::test_photon()
{
    LOG(info); 
    unsigned num_photon = 100 ; 
    std::vector<quad4> photon ; 
    photon.resize(num_photon); 

    qc.generate(   photon.data(), photon.size() ); 
    qc.dump(       photon.data(), photon.size() ); 
    NP::Write( FOLD, "photon.npy" ,  (float*)photon.data(), photon.size(), 4, 4  ); 
}

/**
test_boundary_lookup
----------------------

Does lookups at every point of the 2d float4 boundary texture 

**/

void QCtxTest::test_boundary_lookup()
{
    LOG(info); 
    unsigned width = qc.getBoundaryTexWidth(); 
    unsigned height = qc.getBoundaryTexHeight(); 
    const NPY<float>* src = qc.getBoundaryTexSrc(); 
    unsigned num_lookup = width*height ; 

    std::vector<quad> lookup(num_lookup); 
    qc.boundary_lookup( lookup.data(), width, height ); 

    NP::Write( FOLD, "boundary_lookup.npy" ,  (float*)lookup.data(), height, width, 4 ); 
    src->save( FOLD, "boundary_tex_src.npy" ); 
}

int main(int argc, char** argv)
{
    char test = argc > 1 ? argv[1][0] : 'B' ; 
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
       case 'S': qtc.test_wavelength('S')  ; break ; 
       case 'C': qtc.test_wavelength('C')  ; break ; 
       case 'P': qtc.test_photon();        ; break ; 
       case 'B': qtc.test_boundary_lookup(); break ;  
    }
    return 0 ; 
}
