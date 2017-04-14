
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

    for(unsigned i=0 ; i < 100 ; i++)
    {
        OpticksCSG_t type = (OpticksCSG_t)i ; 
        if(!CSGExists(type)) continue ; 

        const char*  name = CSGName( type );

        std::cout 
                   << " type " << std::setw(3) << type
                   << " name " << std::setw(20) << name
                   << std::endl ; 


       /*
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

        */

    }


    //int rc = SSys::run("tpmt.py");

    return 0 ; 
} 

