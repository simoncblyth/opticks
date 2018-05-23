/*
::

    simon:opticksnpy blyth$ NCSGListTest $TMP/tboolean-torus--
*/

#include "NPY_LOG.hh"
#include "PLOG.hh"
#include "NCSGList.hpp"
#include "NCSG.hpp"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    const char* csgpath = argc > 1 ? argv[1] : NULL ; 
    unsigned verbosity = 0 ; 

    if(csgpath == NULL)
    {
        LOG(warning) << "Expecting 1st argument csgpath directory containing NCSG trees" ; 
        return 0 ;
    } 
   
    NCSGList* ls = NCSGList::Load(csgpath, verbosity );    
    if( ls == NULL )
    {
        LOG(warning) << "FAILED to load NCSG trees from " << csgpath  ; 
        return 0 ;
    }

    ls->dumpDesc();
    ls->dumpMeta();
    ls->dumpUniverse();


    return 0 ; 
}
