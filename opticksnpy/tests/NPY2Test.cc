#include "NPY_FLAGS.hh"

#include <cassert>


// npy-
#include "NGLM.hpp"
#include "NPY.hpp"

#include "BRAP_LOG.hh"
#include "PLOG.hh"



void test_grow()
{
    NPY<float>* onestep = NPY<float>::make(1,2,4);
    onestep->zero();
    float* ss = onestep->getValues()  ; 

    NPY<float>* genstep = NPY<float>::make(0,2,4);

    for(unsigned i=0 ; i < 100 ; i++)
    {
        ss[0*4+0] = float(i) ; 
        ss[0*4+1] = float(i) ; 
        ss[0*4+2] = float(i) ; 
        ss[0*4+3] = float(i) ; 

        ss[1*4+0] = float(i) ; 
        ss[1*4+1] = float(i) ; 
        ss[1*4+2] = float(i) ; 
        ss[1*4+3] = float(i) ; 

        genstep->add(onestep);
    }
    
   genstep->save("$TMP/test_grow.npy");

   LOG(info) << " numItems " << genstep->getNumItems() 
             << " shapeString " << genstep->getShapeString() 
             ;

}



int main(int argc, char** argv )
{
    PLOG_(argc, argv);
    BRAP_LOG_ ;   


    NPYBase::setGlobalVerbose(true);

    test_grow();

    return 0 ; 
}


