/*

NScanTest $TMP/tgltf/extras

*/

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NScan.hpp"
#include "NNode.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  


    const char* basedir = argc > 1 ? argv[1] : NULL ;
    if(!basedir) 
    {
        LOG(warning) << "expecting base directory argument that contains CSG trees" ; 
        return 0 ; 
    }


    int verbosity = 0 ; 
    std::vector<NCSG*> trees ;
    int ntree = NCSG::DeserializeTrees( basedir, trees, verbosity );

    LOG(info) 
          << " NScanTest autoscan all trees " 
          << " basedir " << basedir  
          << " ntree " << ntree 
          << " verbosity " << verbosity 
          ; 


    for(unsigned i=0 ; i < trees.size() ; i++)
    {
        NCSG* csg = trees[i]; 
        nnode* root = csg->getRoot();

        NScan scan(*root, verbosity);
        unsigned nzero = scan.autoscan();

        std::cout 
             << " i " << std::setw(4) << i 
             << " nzero " << std::setw(4) << nzero 
             << " treedir " << std::setw(40) << csg->getTreeDir()  
             << " soname " << std::setw(40) << csg->soname()  
             << " msg " << scan.get_message()
             << std::endl 
             ;
    }

    return 0 ; 
}







