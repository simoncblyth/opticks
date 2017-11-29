#include <iostream>

#include "SSys.hh"
#include "BOpticksResource.hh"
#include "BRAP_LOG.hh"
#include "PLOG.hh"

struct BOpticksResourceTest
{
    BOpticksResourceTest(const char* srcpath, const char* srcdigest)
        :
        _res()
    {
 

        _res.setSrcPathDigest(srcpath, srcdigest);
        _res.Summary();
    }

    BOpticksResource _res ; 

};



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    BRAP_LOG__ ; 

    //const char* srcpath = "/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae" ; 
    //const char* srcdigest = "96ff965744a2f6b78c24e33c80d3a4cd" ; 

    const char* srcpath   = SSys::getenvvar("OPTICKS_SRCPATH");
    const char* srcdigest  = SSys::getenvvar("OPTICKS_SRCDIGEST");

    if(!srcpath) 
    {
        LOG(warning) << " no OPTICKS_SRCPATH envvar " ; 
        return 0 ; 
    }
    if(!srcdigest) 
    {
        LOG(warning) << " no OPTICKS_SRCDIGEST envvar " ; 
        return 0 ; 
    }


    BOpticksResourceTest brt(srcpath, srcdigest) ; 

    const char* treedir_ = brt._res.getDebuggingTreedir(argc, argv);  //  requires the debugging only IDPATH envvar
    std::string treedir = treedir_ ? treedir_ : "/tmp/error-no-IDPATH-envvar" ; 

    std::cout 
              << " treedir " << treedir
              << std::endl 
              ;




    return 0 ; 
}
