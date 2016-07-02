
#include <cassert>
#include "GCSG.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;


    NPY<float>* buffer = NULL ; 
    GItemList* materials = NULL ; 
    GItemList* lvnames = NULL ; 
    GItemList* pvnames = NULL ; 

    GCSG* csg = new GCSG(buffer, materials, lvnames, pvnames );
    assert(csg->getNumItems() == 0 );


    return 0 ;
}

