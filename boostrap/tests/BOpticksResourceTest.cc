#include <iostream>

#include "SSys.hh"
#include "BOpticksResource.hh"
#include "OPTICKS_LOG.hh"

struct BOpticksResourceTest
{
    BOpticksResourceTest(const char* idpath)
        :
        _res()
    {
        _res.setupViaID(idpath);
        _res.Summary();
    }
    
    BOpticksResourceTest(const char* srcpath, const char* srcdigest)
        :
        _res()
    {
        _res.setupViaSrc(srcpath, srcdigest);
        _res.Summary();
    }

    BOpticksResource _res ; 

};



/*

    const char* treedir_ = brt._res.getDebuggingTreedir(argc, argv);  //  requires the debugging only IDPATH envvar
    std::string treedir = treedir_ ? treedir_ : "/tmp/error-no-IDPATH-envvar" ; 

    std::cout 
              << " treedir " << treedir
              << std::endl 
              ;

*/



void test_ViaSrc()
{
    const char* srcpath   = SSys::getenvvar("DEBUG_OPTICKS_SRCPATH");
    const char* srcdigest  = SSys::getenvvar("DEBUG_OPTICKS_SRCDIGEST", "0123456789abcdef0123456789abcdef" );

    assert( srcpath && srcdigest );
 
    BOpticksResourceTest brt(srcpath, srcdigest) ; 
    BOpticksResourceTest brt2(brt._res.getIdPath()) ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    const char* idpath  = SSys::getenvvar("IDPATH");
    if(!idpath) return 0 ;     

    LOG(info) << " starting from IDPATH " << idpath ; 
    BOpticksResourceTest brt(idpath) ; 
    BOpticksResourceTest brt2(brt._res.getSrcPath(), brt._res.getSrcDigest()) ; 

    // the two setup approaches, should yield exactly the same paths 

    return 0 ; 
}
