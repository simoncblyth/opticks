#include <string>
#include <cassert>

#include "BPath.hh"

#include "SYSRAP_LOG.hh"
#include "BRAP_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    SYSRAP_LOG_ ;
    BRAP_LOG_ ;

    const char* idpath_0="/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae" ; 

    BPath p0(idpath_0);
    LOG(info) << p0.desc() ; 

    const char* idpath_1="/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1" ;  
    BPath p1(idpath_1);
    LOG(info) << p1.desc() ; 


    return 0 ; 
}



