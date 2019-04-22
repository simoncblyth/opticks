// TEST=STimeTest om-t

#include <vector>
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
    std::cout << "STime::Format(\"%c\", t) " << STime::Format(t, "%c") << std::endl;
    std::cout << "STime::Format() " << STime::Format() << std::endl;


    std::vector<std::string> fmts = { "%c", "%Y", "%m", "%D", "%d", "%H", "%M", "%Y%m%d_%H%M%S"  } ; 

    for(unsigned i=0 ; i < fmts.size() ; i++)
    {
         std::cout 
               << "STime::Format(0,\"" << fmts[i] << "\") " 
               <<  STime::Format(0, fmts[i].c_str()) 
               << std::endl;
    }



    SSys::run( "date +%s" ); 

    return 0 ; 
}
