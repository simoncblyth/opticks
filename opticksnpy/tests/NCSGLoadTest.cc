/**
Tests individual trees::

    NCSGLoadTest $TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-/1


    NCSGLoadTest 0    
    NCSGLoadTest 66    
        # integer arguments are interpreted as lvidx and NCSG are 
        # loaded from the standard extras dir located within IDFOLD

**/

#include <iostream>

#include "BFile.hh"
#include "BStr.hh"

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NSceneConfig.hpp"
#include "NNode.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


#include "SSys.hh"
#include "BOpticksResource.hh"



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    BOpticksResource okr ;  // no Opticks at this level 
    std::string treedir = okr.getDebuggingTreedir(argc, argv);

    const char* config = NULL ; 
    NCSG* csg = NCSG::LoadCSG( treedir.c_str(), config ); 

    if(!csg) return 0 ; 

    return 0 ; 
}


