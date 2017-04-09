#include "NPY_FLAGS.hh"

#include <cassert>


// npy-
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "NPY.hpp"
#include "GLMFormat.hpp"

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

        transforms->setMat4( m, i, -1, false ); // dont transpose  
    }
    transforms->save("$TMP/test_make_inverted_transforms/transforms.npy");


    NPY<float>* itransforms = NPY<float>::make_inverted_transforms( transforms, false );
    NPY<float>* itransforms_T = NPY<float>::make_inverted_transforms( transforms, true );

    itransforms->save("$TMP/test_make_inverted_transforms/itransforms.npy");
    itransforms_T->save("$TMP/test_make_inverted_transforms/itransforms_T.npy");
}





NPY<float>* make_src_transforms()
{
    NPY<float>* src = NPY<float>::make(3,4,4) ; 
    src->zero();

    float angle = 45.f ; 
    for(unsigned i=0 ; i < 3 ; i++)
    {
        glm::vec3 axis( i == 0 ? 1 : 0 , i == 1 ? 1 : 0, i == 2 ? 1 : 0 );
        glm::vec3 tlat = axis*100.f ; 

        glm::mat4 tr(1.f);
        tr = glm::rotate(tr, angle*glm::pi<float>()/180.f, axis ) ;
        tr = glm::translate(tr, tlat );
 
        src->setMat4( tr, i); 
    }
    return src ; 
}


void test_Mat4Pairs()
{
    NPY<float>* src = make_src_transforms();
    assert(src->hasShape(3,4,4));

    NPY<float>* transforms = NPY<float>::make(3,2,4,4) ; 
    transforms->zero();

    NPY<float>* transforms1 = NPY<float>::make(6,4,4) ; 
    transforms1->zero();

    for(unsigned i=0 ; i < 3 ; i++)
    {
        glm::mat4 tr = src->getMat4(i);
        glm::mat4 irit = invert_tr(tr);

        transforms->setMat4( tr  , i, 0 ); 
        transforms->setMat4( irit, i, 1 ); 

        transforms1->setMat4( tr   , i*2 + 0 ); 
        transforms1->setMat4( irit , i*2 + 1 ); 
    }
    transforms->save("$TMP/test_Mat4Pairs/transforms.npy");
    transforms1->save("$TMP/test_Mat4Pairs/transforms1.npy");


    for(int i=0 ; i < 3 ; i++)
    {
        for(int j=0 ; j < 2 ; j++)
        {
            glm::mat4 mat = transforms->getMat4(i,j);
            glm::mat4 mat1 = transforms1->getMat4(2*i+j);
            assert( mat == mat1 );

            std::cout << "(" << i << "," << j << ")" 
                      << std::endl
                      << gpresent( "mat", mat )
                      << std::endl
                      ;
          
        }
    }
}


void test_make_paired_transforms()
{
    NPY<float>* src = make_src_transforms();
    assert(src->hasShape(3,4,4));

    NPY<float>* paired = NPY<float>::make_paired_transforms(src);
    assert(paired->hasShape(3,2,4,4));
    paired->save("$TMP/test_make_paired_transforms/transforms.npy");


    NPY<float>* empty = NPY<float>::make(0, 4, 4);
    empty->zero();
 
    NPY<float>* epaired = NPY<float>::make_paired_transforms(empty);
    assert(epaired->hasShape(0,2,4,4));
    epaired->save("$TMP/test_make_paired_transforms/epaired.npy");

   // hmm again, refuses to save an empty 

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
    //test_make_identity_transforms();

    test_Mat4Pairs();
    test_make_paired_transforms();

    return 0 ; 
}


