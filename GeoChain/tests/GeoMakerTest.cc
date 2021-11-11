#include "OPTICKS_LOG.hh"
#include "GeoMaker.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    bool default_ = GeoMaker::CanMake("default") ; 
    assert( default_ == true ); 

    bool other = GeoMaker::CanMake("other") ; 
    assert( other == false ); 

    return 0 ; 
}
