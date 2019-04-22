// TEST=STimeTest om-t

#include <iostream>

#include "STime.hh"
#include "SSys.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv) 
{
    OPTICKS_LOG(argc, argv);

    LOG(info); 

    int t = STime::EpochSeconds() ;      

    std::cout << "STime::EpochSeconds() " << t << std::endl ; 
    std::cout << "STime::Format(\"%c\", t) " << STime::Format("%c", t ) << std::endl;
    std::cout << "STime::Format() " << STime::Format() << std::endl;

    SSys::run( "date +%s" ); 

    return 0 ; 
}
