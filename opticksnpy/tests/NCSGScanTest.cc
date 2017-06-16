/**
Tests individual trees::

    NCSGLoadTest $TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-/1

**/

#include <iostream>

#include "BFile.hh"
#include "BStr.hh"


#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    LOG(info) << " argc " << argc << " argv[0] " << argv[0] ;  

    const char* treedir = argc > 1 ? argv[1] : "$TMP/csg_py/1"  ;

    if(!BFile::ExistsDir(treedir))
    {
         LOG(warning) << argv[0] << " no such dir " << treedir ;
         return 0 ; 
    }
 
    int verbosity = 2 ; 
    NCSG* csg = NCSG::LoadTree(treedir, verbosity );
    assert(csg);




    const char* fallback = "0,0,128,0,0,1,-1,1,0.001" ; 

    std::vector<float> f ; 
    BStr::fsplitEnv(f, "SCAN", fallback, ',' );

    bool has9 = f.size() == 9 ;
    if(!has9) LOG(fatal) << "NCSGScan"
                         << " SCAN envvar required 9 comma delimited elements" 
                         << " got " << f.size()
                        ;
    assert(has9);



    nnode* root = csg->getRoot();

    glm::vec3 origin(    f[0],f[1],f[2] );
    glm::vec3 direction( f[3],f[4],f[5] );
    glm::vec3 range(     f[6],f[7],f[8] );

    std::vector<float> sd ; 
    bool dump = true ; 

    nnode::Scan(sd, *root, origin, direction, range, dump );




    return 0 ; 
}


