
#include <cstring>
#include <iostream>
#include <iomanip>

#include "SYSRAP_LOG.hh"
#include "PLOG.hh"

#include "OpticksCSG.h"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    SYSRAP_LOG__ ; 

    for(unsigned i=1 ; i < CSG_UNDEFINED ; i++)
    {

        OpticksCSG_t csgFlag = (OpticksCSG_t)i ; 

        const char* csgName = CSGName( csgFlag );
        char csgChar = CSGChar( csgName );

        OpticksCSG_t csgFlag2 = CSGFlag(csgChar);
        assert( csgFlag2 == csgFlag );

        const char* csgName2 = CSGChar2Name( csgChar );

        assert(strcmp(csgName, csgName2) == 0);
        assert( csgName == csgName2 );

        std::cout 
                   << " csgFlag " << std::setw(3) << csgFlag 
                   << " csgName " << std::setw(20) << csgName
                   << " csgChar " << std::setw(3) << csgChar
                   << " csgFlag2 " << std::setw(3) << csgFlag2
                   << " csgName2 " << std::setw(20) << csgName2
                   << std::endl ; 
    }

    //int rc = SSys::run("tpmt.py");

    return 0 ; 
} 

