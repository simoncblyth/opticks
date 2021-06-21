#include "Opticks.hh"
#include "SPath.hh"
#include "NP.hh"
#include "GScintillatorLib.hh"

#include "QRng.hh"
#include "QCtx.hh"
#include "scuda.h"

#include "OPTICKS_LOG.hh"


void test_wavelength(QCtx& qc)
{
    unsigned num_wavelength = 100000 ; 

    std::vector<float> wavelength ; 
    wavelength.resize(num_wavelength, 0.f); 

    qc.generate(   wavelength.data(), wavelength.size() ); 
    qc.dump(       wavelength.data(), wavelength.size() ); 

    NP::Write( "/tmp/QCtxTest", "wavelength.npy" ,  wavelength ); 
}



void test_photon(QCtx& qc)
{
    unsigned num_photon = 100 ; 

    std::vector<quad4> photon ; 
    photon.resize(num_photon); 

    qc.generate(   photon.data(), photon.size() ); 
    qc.dump(       photon.data(), photon.size() ); 

    NP::Write( "/tmp/QCtxTest", "photon.npy" ,  (float*)photon.data(), photon.size(), 4, 4  ); 
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

    QCtx qc(slib);     // uploads reemission texture  

    test_wavelength(qc); 
    test_photon(qc); 

    return 0 ; 
}

