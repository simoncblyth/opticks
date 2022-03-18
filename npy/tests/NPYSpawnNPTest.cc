#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "NP.hh"

int main(int argc, char** argv)
{ 
    OPTICKS_LOG(argc, argv); 

    double domain_low = 100. ; 
    double domain_high = 200. ; 
    std::string creator = argv[0] ; 


    NPY<double>* wav = NPY<double>::make(4, 4) ; 
    wav->zero(); 
    wav->dump(); 
    wav->setMeta("domain_low",   domain_low );
    wav->setMeta("domain_high",   domain_high );
    wav->setMeta<std::string>("creator",  creator ); 

    NP* wv = wav->spawn(); 
   
    LOG(info) << "wv.meta [" << std::endl << wv->meta << "]"   ; 


    return 0 ; 
}
