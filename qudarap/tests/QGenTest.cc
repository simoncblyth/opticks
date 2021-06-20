
#include <vector>

#include "QRng.hh"
#include "QGen.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    QRng rng ;   // loads and uploads curandState 
    LOG(info) << rng.desc(); 

    std::vector<float> uu ; 
    uu.resize(100, 0.); 

    QGen gen ; 
    gen.generate(uu.data(), uu.size() ); 
    gen.dump( uu.data(), uu.size()); 
  
    return 0 ; 

}
