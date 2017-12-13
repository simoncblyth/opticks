
#include "SDirect.hh"
#include <iomanip>
#include <iostream>

#include "SSys.hh"
#include "SYSRAP_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 

    LOG(info) << argv[0] ; 

    SSys::Dump(argv[0]); 

    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {
        cout_redirect out_(coutbuf.rdbuf());
        cerr_redirect err_(cerrbuf.rdbuf());
        
        SSys::Dump(argv[0]); 
        
        // dtors of the redirect structs reset back to standard cout/cerr streams  
    }        

    std::string out = coutbuf.str(); 
    std::string err = cerrbuf.str(); 

    LOG(info) << " captured cout " << out.size()  ; 
    std::cout << "[" << std::endl << out << "]" << std::endl  ; 

    LOG(info) << " captured cerr " << err.size() ; 
    std::cout << "[" << std::endl << err << "]" << std::endl  ; 



    SSys::Dump(argv[0]); 


    return 0 ; 
}

