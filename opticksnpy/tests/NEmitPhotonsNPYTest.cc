/*
NEmitPhotonsNPYTest $TMP/tboolean-torus--
*/

#include <cstdlib>
#include <cfloat>

#include "SSys.hh"
#include "NGLMExt.hpp"

#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "BRAP_LOG.hh"
#include "SYSRAP_LOG.hh"

#include "NCSGList.hpp"
#include "NEmitPhotonsNPY.hpp"
#include "NPho.hpp"
#include "NPY.hpp"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    SYSRAP_LOG__ ;  
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

    unsigned EMITSOURCE = 0x1 << 18 ; 
    unsigned gencode = EMITSOURCE ; 

    NEmitPhotonsNPY ep(csg, gencode) ;

    NPY<float>* ox = ep.getPhotons();
    ox->dump();

    NPY<float>* gs = ep.getFabStepData();
    gs->dump();


    const char* path = "$TMP/NEmitPhotonsNPYTest_fabstep.npy" ;
    gs->save(path);
    SSys::npdump(path, "np.int32");


    NPho ph(ox) ;
    unsigned modulo = 10000 ; 
    unsigned margin = 10 ; 
    ph.dump(modulo, margin); 


    return 0 ; 
}


