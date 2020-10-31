#include "OPTICKS_LOG.hh"
#include "SSys.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
   
    const char* hostname = SSys::hostname() ; 
    const char* username = SSys::username() ; 
    LOG(info) 
       << " hostname " << hostname 
       << " username " << username 
       ;

    assert( hostname ); 
    assert( username ); 

    return 0 ; 
}
