#include <iostream>

#include "NPY.hpp"
#include "NCSG.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    const char* base = argc > 0 ? argv[1] : "$TMP/csg_py" ; 
    LOG(info) << argv[0] << " NCSG::Deserialize directory " << base ;  

    std::vector<NCSG*> trees ;
    NCSG::Deserialize( base, trees );

    LOG(info) << "NCSG::Deserialize found trees : " << trees.size() ;
    for(unsigned i=0 ; i < trees.size() ; i++) trees[i]->dump("NCSGTest dump");

    return 0 ; 
}


