#include <cassert>

#include "NPY.hpp"
#include "NLoad.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"

int main(int argc, char** argv)
{
     PLOG_(argc, argv);
     NPY_LOG__ ; 

     NPYBase::setGlobalVerbose(true);

     NPY<float>* gs_0 = NLoad::Gensteps("dayabay","cerenkov","1") ;
     NPY<float>* gs_1 = NLoad::Gensteps("juno",   "cerenkov","1") ;
     NPY<float>* gs_2 = NLoad::Gensteps("dayabay","scintillation","1") ;
     NPY<float>* gs_3 = NLoad::Gensteps("juno",   "scintillation","1") ;

     assert(gs_0);
     assert(gs_1);
     assert(gs_2);
     assert(gs_3);

     //gs_0->dump();
     //gs_1->dump();
     //gs_2->dump();
     //gs_3->dump();
 
     return 0 ; 
}
