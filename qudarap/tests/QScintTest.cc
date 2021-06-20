#include "Opticks.hh"
#include "SPath.hh"
#include "NP.hh"
#include "GScintillatorLib.hh"

#include "QRng.hh"
#include "QScint.hh"
#include "scuda.h"

#include "OPTICKS_LOG.hh"


void test_wavelength(QScint& sc)
{
    unsigned num_wavelength = 100000 ; 

    std::vector<float> wavelength ; 
    wavelength.resize(num_wavelength, 0.f); 

    sc.generate(   wavelength.data(), wavelength.size() ); 
    sc.dump(       wavelength.data(), wavelength.size() ); 

    NP::Write( "/tmp/QScintTest", "wavelength.npy" ,  wavelength ); 
}



void test_photon(QScint& sc)
{
    unsigned num_photon = 100 ; 

    std::vector<quad4> photon ; 
    photon.resize(num_photon); 

    sc.generate(   photon.data(), photon.size() ); 
    sc.dump(       photon.data(), photon.size() ); 

    NP::Write( "/tmp/QScintTest", "photon.npy" ,  (float*)photon.data(), photon.size(), 4, 4  ); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    GScintillatorLib* slib = GScintillatorLib::load(&ok);
    slib->dump();

    QRng rng ;             // loads and uploads curandState 
    LOG(info) << rng.desc(); 

    QScint sc(slib);     // uploads reemission texture  

    test_wavelength(sc); 
    test_photon(sc); 

    return 0 ; 
}

