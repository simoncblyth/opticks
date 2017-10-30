/*


::

    simon:opticksnpy blyth$ NCSGListTest $TMP/tboolean-torus--
    2017-10-30 15:56:04.541 INFO  [1594433] [NCSG::Deserialize@1197] NCSG::Deserialize VERBOSITY 0 basedir /tmp/blyth/opticks/tboolean-torus-- txtpath /tmp/blyth/opticks/tboolean-torus--/csg.txt nbnd 2
    2017-10-30 15:56:04.543 INFO  [1594433] [main@32]  numTrees 2
    simon:opticksnpy blyth$ 


*/


#include "NPY_LOG.hh"
#include "PLOG.hh"
#include "NCSGList.hpp"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    const char* csgpath = argc > 1 ? argv[1] : NULL ; 
    if(csgpath == NULL)
    {
        LOG(warning) << "Expecting 1st argument csgpath directory containing NCSG trees" ; 
        return 0 ;
    } 

    unsigned verbosity = 0 ; 
    NCSGList trees(csgpath, verbosity );    

    unsigned numTrees = trees.getNumTrees() ;
    LOG(info) << " numTrees " << numTrees ; 



    LOG(info) << " meta \n" ; 
    for(unsigned i=0 ; i < numTrees ; i++)
    {
        NCSG* tree = trees.getTree(i) ;
        std::cout << " tree " << std::setw(2) << i 
                  << " meta " << tree->meta()
                  << std::endl ;  
    }


    LOG(info) << " desc \n" ; 
    for(unsigned i=0 ; i < numTrees ; i++)
    {
        NCSG* tree = trees.getTree(i) ;
        std::cout << " tree " << std::setw(2) << i 
                  << " desc " << tree->desc()
                  << std::endl ;  
    }







    return 0 ; 
}
