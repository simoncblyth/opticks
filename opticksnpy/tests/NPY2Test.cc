#include "NPY_FLAGS.hh"

#include <cassert>


// npy-
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "NPY.hpp"

#include "NPY_LOG.hh"
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


void test_make_inverted_transforms()
{
    NPY<float>* transforms = NPY<float>::make(3, 4, 4);
    transforms->zero();

    float angle = 45.f ; 
    for(unsigned i=0 ; i < 3 ; i++)
    {
        glm::vec3 axis( i == 0 ? 1 : 0 , i == 1 ? 1 : 0, i == 2 ? 1 : 0 );
        glm::vec3 tlat = axis*100.f ; 

        glm::mat4 m(1.f);
        m = glm::rotate(m, angle*glm::pi<float>()/180.f, axis ) ;
        m = glm::translate(m, tlat );

        transforms->setMat4( m, i, false ); // dont transpose  
    }
    transforms->save("$TMP/test_make_inverted_transforms/transforms.npy");


    NPY<float>* itransforms = NPY<float>::make_inverted_transforms( transforms, false );
    NPY<float>* itransforms_T = NPY<float>::make_inverted_transforms( transforms, true );

    itransforms->save("$TMP/test_make_inverted_transforms/itransforms.npy");
    itransforms_T->save("$TMP/test_make_inverted_transforms/itransforms_T.npy");
}


void test_make_inverted_transforms_empty()
{
   // with an empty buffer get buffer not allocated warning and nothing gets saved

    NPY<float>* empty = NPY<float>::make(0, 4, 4);
    empty->zero();
    NPY<float>* iempty = NPY<float>::make_inverted_transforms( empty, false );
    empty->save("$TMP/test_make_inverted_transforms/empty.npy");
    iempty->save("$TMP/test_make_inverted_transforms/iempty.npy");
}

void test_make_identity_transforms()
{
    NPY<float>* ids = NPY<float>::make_identity_transforms(10) ;
    ids->save("$TMP/test_make_identity_transforms/ids.npy");
}


int main(int argc, char** argv )
{
    PLOG_(argc, argv);
    NPY_LOG__ ;   
    BRAP_LOG_ ;   


    NPYBase::setGlobalVerbose(true);

    //test_grow();
    //test_make_inverted_transforms();
    //test_make_inverted_transforms_empty();
    test_make_identity_transforms();

    return 0 ; 
}


