// TEST=OpticksCfg2Test om-t


#include "Opticks.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ;

    Opticks* ok = new Opticks(argc, argv);

    ok->configure();


    const char* csgskiplv = ok->getCSGSkipLV(); 
    LOG(info) << " csgskiplv " << ( csgskiplv ? csgskiplv  : "NONE" )  ; 


    LOG(info) << "DONE "  ;



    return 0 ; 
}
