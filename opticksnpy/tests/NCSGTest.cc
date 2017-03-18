#include <iostream>

#include "NPY.hpp"
#include "NCSG.hpp"

#include "PLOG.hh"
#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    const char* base = "$TMP/csg_py" ; 

    std::vector<NCSG*> trees ;

    NCSG::Deserialize( base, trees );

    LOG(info) << "Deserialize " << trees.size() ;

    return 0 ; 
}



