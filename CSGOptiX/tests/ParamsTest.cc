
#include "OPTICKS_LOG.hh"
#include <iostream>
#include "Params.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    std::cout << "sizeof(Params)" << sizeof(Params) << std::endl ;

    LOG(none)    << "sizeof(Params)" << sizeof(Params) ;
    LOG(fatal)   << "sizeof(Params)" << sizeof(Params) ;
    LOG(error)   << "sizeof(Params)" << sizeof(Params) ;
    LOG(warning) << "sizeof(Params)" << sizeof(Params) ;
    LOG(info)    << "sizeof(Params)" << sizeof(Params) ;
    LOG(debug)   << "sizeof(Params)" << sizeof(Params) ;
    LOG(verbose) << "sizeof(Params)" << sizeof(Params) ;


    return 0 ;
}
