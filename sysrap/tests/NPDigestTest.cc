#include "OPTICKS_LOG.hh"
#include "NP.hh"
#include "SDigest.hh"
#include "NPDigest.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* h = "hello" ; 
    std::string xdig = SDigest::Buffer( h, 5 ); 

    std::cout 
        << " i " << std::setw(4) << -1 
        << " xdig " << xdig 
        << std::endl
        ; 


    NP* a = NP::Make<char>(3,5) ;
    char* aa = a->values<char>(); 
    memcpy( aa+0  , h, 5 );  
    memcpy( aa+5  , h, 5 );  
    memcpy( aa+10 , h, 5 );  

    for(int i=0 ; i < a->shape[0] ; i++)
    {
        std::string dig = NPDigest::ArrayItem( a, i ); 
        std::cout 
            << " i " << std::setw(4) << i 
            << "  dig " << dig
            << std::endl 
            ;

        assert( strcmp( dig.c_str(), xdig.c_str() ) == 0 ); 
    }

    return 0 ; 
}


