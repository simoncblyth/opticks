/*
NEmitPhotonsNPYTest $TMP/tboolean-torus--
*/

#include <cstdlib>
#include <cfloat>

#include "NGLMExt.hpp"

#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "BRAP_LOG.hh"

#include "NCSGList.hpp"
#include "NEmitPhotonsNPY.hpp"
#include "NPY.hpp"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  
    BRAP_LOG__ ;  

    const char* csgpath = argc > 1 ? argv[1] : NULL ; 
    if(csgpath == NULL)
    {
        LOG(warning) << "Expecting 1st argument csgpath directory containing NCSG trees" ; 
        return 0 ;
    } 

    unsigned verbosity = 0 ; 
    NCSGList trees(csgpath, verbosity );    

    NCSG* csg = trees.findEmitter();

    unsigned numTrees = trees.getNumTrees() ;
    LOG(info) << " numTrees " << numTrees 
              << " emitter " << csg
              ; 

    if(csg == NULL)
    {
        LOG(warning) << " failed to find emitter in those trees " ; 
        return 0 ; 
    }

    NEmitPhotonsNPY ep(csg) ;

    NPY<float>* ox = ep.getNPY();
    ox->dump();




    return 0 ; 
}


