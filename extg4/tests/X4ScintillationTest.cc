/**
X4ScintillationTest
=============================

1. loads /tmp/G4OpticksAnaMgr/FASTCOMPONENT.npy 
2. performs the numerical integration just like DsG4Scintillation with X4Scintillation
3. converts to X4Array to allow persisting
4. persists to /tmp/G4OpticksAnaMgr/X4ScintillationTest.npy

::

    diff ScintillationIntegral.npy X4ScintillationTest.npy

**/

#include "Opticks.hh"

#include "X4MaterialPropertyVector.hh"
#include "X4Array.hh"
#include "X4Scintillation.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include "OPTICKS_LOG.hh"
#include "NPY.hpp"


void test_manual(const char* outdir, const NPY<double>* slow_en, const NPY<double>* fast_en, const NPY<double>* THE_buffer  )
{
    LOG(info) << "[" ; 
    G4MaterialPropertyVector* theFastLightVector = X4MaterialPropertyVector::FromArray(fast_en) ; 
    G4MaterialPropertyVector* ScintillatorIntegral = X4Scintillation::Integral(theFastLightVector) ; 

    double e0 = ScintillatorIntegral->GetMaxLowEdgeEnergy();
    double e1 = ScintillatorIntegral->GetMinLowEdgeEnergy();
    double v0 = ScintillatorIntegral->GetMinValue() ;  
    double v1 = ScintillatorIntegral->GetMaxValue() ;  

    NPY<double>* meta = NPY<double>::make(5); 
    meta->zero(); 
    int j = 0 ; 
    int k = 0 ; 
    int l = 0 ; 
    meta->setValue(0,j,k,l, e0 ); 
    meta->setValue(1,j,k,l, e1 ); 
    meta->setValue(2,j,k,l, v0); 
    meta->setValue(3,j,k,l, v1); 
    meta->save(outdir, "meta.npy");  

    NPY<double>* si = X4Array::Convert<double>(ScintillatorIntegral) ; 
    const char* derived_name = "ScintillatorIntegral.npy" ; 
    LOG(info) << " save to " << outdir << "/" << derived_name ; 
    si->save(outdir, derived_name ); 

    NPY<double>* wl = X4Scintillation::CreateWavelengthSamples( ScintillatorIntegral, 1000000 ); 
    const char* localSamples = "g4localSamples.npy" ; 
    LOG(info) << " save to " << outdir << "/" << localSamples ; 
    wl->save(outdir, localSamples); 

    unsigned num_bins = 4096 ; 
    unsigned hd_factor = 20 ; 
    const char* name = "LS" ; 
    bool energy_not_wavelength = false ; 

    NPY<double>* g4icdf = X4Scintillation::CreateGeant4InterpolatedInverseCDF(ScintillatorIntegral, num_bins, hd_factor, name, energy_not_wavelength ) ; 

    LOG(info) << " compare THE_buffer and g4icdf with various epsilon " ; 
    std::vector<double> epsilons = { 1e-6 , 1e-4, 1e-3 } ; // fails for 1e-8 from CLHEP h_Planck change
    bool dump = true ; 
    unsigned dumplimit = 100 ; 
    char mode = 'R' ; 
    unsigned mismatch_tot = NPY<double>::compare( THE_buffer, g4icdf, epsilons, dump, dumplimit, mode ); 
    LOG(info) << " mismatch_tot " << mismatch_tot ; 
    assert( mismatch_tot == 0 );  

    const char* g4icdf_name = "g4icdf_manual.npy" ; 
    LOG(info) << " save to " << outdir << "/" << g4icdf_name ; 
    g4icdf->save(outdir, g4icdf_name);  



    energy_not_wavelength = true ; 
    NPY<double>* g4icdf_energy = X4Scintillation::CreateGeant4InterpolatedInverseCDF(ScintillatorIntegral, num_bins, hd_factor, name, energy_not_wavelength ) ; 
    const char* g4icdf_energy_name = "g4icdf_energy_manual.npy" ; 
    g4icdf_energy->save(outdir, g4icdf_energy_name );  

    LOG(info) << "]" ; 
}

void test_auto(const char* outdir, const NPY<double>* slow_en, const NPY<double>* fast_en, const NPY<double>* THE_buffer  )
{
    LOG(info) << "[" ; 
    X4Scintillation xs(slow_en, fast_en); 
    NPY<double>* g4icdf = xs.createGeant4InterpolatedInverseCDF() ; 

    const char* name = "g4icdf_auto.npy" ;  
    LOG(info) << " save to " << outdir << "/" << name ; 
    g4icdf->save(outdir, name); 

    LOG(info) << " compare THE_buffer and g4icdf with various epsilon " ; 
    std::vector<double> epsilons = { 1e-6 , 1e-4, 1e-3 } ; 
    bool dump = true ; 
    unsigned dumplimit = 100 ; 
    char mode = 'R' ; 
    unsigned mismatch_tot = NPY<double>::compare( THE_buffer, g4icdf, epsilons, dump, dumplimit, mode ); 
    LOG(info) << " mismatch_tot " << mismatch_tot ; 
    assert( mismatch_tot == 0 );  
    LOG(info) << "]" ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 
    const char* keydir = ok.getKeyDir(); 
 
    // TODO: special $KEYDIR internal var interpretation by path handling, with fallback to envvar OPTICKS_KEYDIR
    LOG(info) << " load slow_en/fast_en from keydir/GScintillatorLib/LS_ori/  " << keydir ; 
    NPY<double>* slow_en = NPY<double>::load(keydir, "GScintillatorLib/LS_ori/SLOWCOMPONENT.npy" ); 
    NPY<double>* fast_en = NPY<double>::load(keydir, "GScintillatorLib/LS_ori/FASTCOMPONENT.npy" ); 
    NPY<double>* THE_buffer = NPY<double>::load(keydir, "GScintillatorLib/GScintillatorLib.npy" ); 

    if( slow_en == nullptr || fast_en == nullptr ) return 0 ; 

    const char* outdir = "$TMP/X4ScintillationTest" ; 
    test_manual(outdir, slow_en, fast_en, THE_buffer ); 
    test_auto(outdir, slow_en, fast_en, THE_buffer ); 

    THE_buffer->save(outdir, "GScintillatorLib.npy" ); // duplicate for easy access from tests/X4ScintillationTest.py 

    return 0 ; 
}
