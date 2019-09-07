/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

// TEST=NPY2Test om-t

#include "NPY_FLAGS.hh"

#include <cassert>


// npy-
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "NPY.hpp"
#include "GLMFormat.hpp"

#include "OPTICKS_LOG.hh"



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





NPY<float>* make_src_transforms(unsigned rep)
{
    NPY<float>* src = NPY<float>::make(3*rep,4,4) ; 
    src->zero();

    float angle = 45.f ; 

    for(unsigned r=0 ; r < rep ; r++)
    {
        for(unsigned i=0 ; i < 3 ; i++)
        {
            glm::vec3 axis( i == 0 ? 1 : 0 , i == 1 ? 1 : 0, i == 2 ? 1 : 0 );
            glm::vec3 tlat = axis*100.f ; 

            glm::mat4 tr(1.f);
            tr = glm::rotate(tr, angle*glm::pi<float>()/180.f, axis ) ;
            tr = glm::translate(tr, tlat );
     
            src->setMat4( tr, r*3+i ); 
        }
    }
    return src ; 
}


void test_Mat4Pairs()
{
    NPY<float>* src = make_src_transforms(1);
    assert(src->hasShape(3,4,4));

    NPY<float>* transforms = NPY<float>::make(3,2,4,4) ; 
    transforms->zero();

    NPY<float>* transforms1 = NPY<float>::make(6,4,4) ; 
    transforms1->zero();

    for(unsigned i=0 ; i < 3 ; i++)
    {
        glm::mat4 tr = src->getMat4(i);
        glm::mat4 irit = nglmext::invert_tr(tr);

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
    NPY<float>* src = make_src_transforms(1);
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


void test_save_load_empty()
{
    NPY<float>* empty = NPY<float>::make(0, 4, 4);
    empty->zero();
    const char* path = "$TMP/npy/test_save_load_empty.npy" ; 
    empty->save(path); 

    NPY<float>* empty2 = NPY<float>::load(path); 
    assert( empty2 ); 

    LOG(info) 
        << " empty " << empty->getShapeString() 
        << " empty2 " << empty2->getShapeString() 
        ;

    assert( empty->equals(empty2,true) ) ; 
    assert( empty2->equals(empty,true) ) ; 
}

/*
epsilon:tests blyth$ xxd /tmp/blyth/opticks/test_save_empty.npy
00000000: 934e 554d 5059 0100 4600 7b27 6465 7363  .NUMPY..F.{'desc
00000010: 7227 3a20 273c 6634 272c 2027 666f 7274  r': '<f4', 'fort
00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
00000030: 652c 2027 7368 6170 6527 3a20 2830 2c20  e, 'shape': (0, 
00000040: 342c 2034 292c 207d 2020 2020 2020 200a  4, 4), }       .

In [1]: a = np.load("/tmp/blyth/opticks/test_save_empty.npy")

In [2]: a
Out[2]: array([], shape=(0, 4, 4), dtype=float32)

*/





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


void test_getItemDigestString()
{
    LOG(info) << "test_getItemDigestString" ; 

    NPY<float>* src = make_src_transforms(2);
    assert(src->hasShape(6,4,4));

    for(unsigned i=0 ; i < 6 ; i++)
         std::cout << " src " << i << " " << src->getItemDigestString(i) << std::endl ; 
 
    NPY<float>* paired = NPY<float>::make_paired_transforms(src);
    assert(paired->hasShape(6,2,4,4));
 
    for(unsigned i=0 ; i < 6 ; i++)
         std::cout << " pai " << i << " " << paired->getItemDigestString(i) << std::endl ; 
 
}


void test_addItemUnique()
{
    LOG(info) << "test_addItemUnique" ; 

    NPY<float>* src = make_src_transforms(2);
    assert(src->hasShape(6,4,4));

    for(unsigned i=0 ; i < 6 ; i++)
         std::cout << " src " << i << " " << src->getItemDigestString(i) << std::endl ; 
 
    NPY<float>* paired = NPY<float>::make_paired_transforms(src);
    assert(paired->hasShape(6,2,4,4));
     
    unsigned ni = paired->getNumItems();
    assert(ni == 6);


    NPY<float>* uniq = NPY<float>::make(0,2,4,4);

    for(unsigned i=0 ; i < ni ; i++)
    {
        unsigned uniq_i = uniq->addItemUnique( paired, i );

        std::cout 
              << " i " << std::setw(2) << i 
              << " uniq_i " << std::setw(2) << uniq_i
              << std::endl  ;

    }
    assert( uniq->getNumItems() == 3 );
    uniq->save("$TMP/test_addItemUnique/uniq.npy");


}



void test_uint32()
{
    NPY<unsigned>* u = NPY<unsigned>::load("$TMP/c.npy") ;
    if(!u) return ; 

    u->dump();

    std::vector<unsigned> vec ; 
    u->copyTo(vec);

    LOG(info) << "loaded " << vec.size() ; 

    assert( vec.size() == u->getNumItems());

}



void test_copyTo_ivec4()
{
    NPY<int>* faces_ = NPY<int>::load("$TMP/tboolean-prism--/1/srcfaces.npy" ) ;
    if(!faces_) return ; 

    faces_->dump();

    std::vector<glm::ivec4> faces ;  
    faces_->copyTo(faces); 
    assert( faces.size() == faces_->getShape(0) ); 

    for(unsigned i=0 ; i < faces.size() ; i++) std::cout << gpresent(faces[i]) << std::endl ; 

}

void test_copyTo_vec3()
{
    NPY<float>* verts_ = NPY<float>::load("$TMP/tboolean-prism--/1/srcverts.npy" ) ;
    if(!verts_) return ; 
    verts_->dump();

    std::vector<glm::vec3> verts ;  
    verts_->copyTo(verts); 
    assert( verts.size() == verts_->getShape(0) ); 

    for(unsigned i=0 ; i < verts.size() ; i++) std::cout << gpresent(verts[i]) << std::endl ; 

}
 
void test_make_from_str()
{
    NPY<unsigned>* u = NPY<unsigned>::make_from_str("1,3,5,7,9") ; 
    u->dump();
    assert( u->getShape(0) == 5 );
}

void test_make_masked()
{
    unsigned N = 10 ; 
    NPY<float>* src = NPY<float>::make(N,4) ; 
    src->zero();

    glm::vec4 v ; 
    for(unsigned i=0 ; i < N ; i++) 
    {
        v.x = float(i*100+0) ; 
        v.y = float(i*100+1) ; 
        v.z = float(i*100+2) ; 
        v.w = float(i*100+3) ; 
        src->setQuad(v, i ); 
    }
    src->dump("src");


    NPY<unsigned>* msk = NPY<unsigned>::make_from_str("1,3,5,7,9") ; 
    unsigned ni_msk = msk->getShape(0); 
    msk->dump("msk");

    NPY<float>* dst = NPY<float>::make_masked( src, msk );
    unsigned ni_dst = dst->getShape(0); 
    dst->dump("dst");

    assert( ni_msk == ni_dst );
}





int main(int argc, char** argv )
{
    OPTICKS_LOG(argc, argv);

    NPYBase::setGlobalVerbose(true);

/*
    test_grow();
    test_make_inverted_transforms();
    test_make_inverted_transforms_empty();
    test_make_identity_transforms();

    test_Mat4Pairs();
    test_make_paired_transforms();

    test_getItemDigestString();
    test_addItemUnique();
    test_uint32();
    test_copyTo_ivec4();
    test_copyTo_vec3();
    test_make_from_str();
    test_make_masked();
*/
    test_save_load_empty();


    return 0 ; 
}


