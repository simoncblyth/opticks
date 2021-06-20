#include "Opticks.hh"
#include "SPath.hh"
#include "NP.hh"
#include "GScintillatorLib.hh"

#include "QRng.hh"
#include "QScint.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    GScintillatorLib* slib = GScintillatorLib::load(&ok);
    slib->dump();

    QRng rng ;             // loads and uploads curandState 
    LOG(info) << rng.desc(); 

    QScint qsc(slib);     // uploads reemission texture  

    unsigned num_wavelength = 100000 ; 

    std::vector<float> wavelength ; 
    wavelength.resize(num_wavelength, 0.f); 

    qsc.generate( wavelength.data(), wavelength.size() ); 
    qsc.dump(     wavelength.data(), wavelength.size() ); 

    NP::Write( "/tmp", "QScintTest.npy" ,  wavelength ); 

    return 0 ; 
}

