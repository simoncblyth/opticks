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

// TEST=NBox2Test om-t


#include <cstdlib>
#include <cfloat>

#include "NBox.hpp"
#include "NGLMExt.hpp"
#include "nmat4triple.hpp"
#include "NBBox.hpp"

#include "OPTICKS_LOG.hh"

void test_resizeToFit()
{
    float h = 10.f ; 
    nbox* box = nbox::Create(2*h,2*h,2*h,0, CSG_BOX3); 

    box->verbosity = 3 ;  
    box->pdump("original nbox::Create(2*h,2*h,2*h,0,CSG_BOX3)");

    nbbox bb = box->bbox_model();

    LOG(info) << " bb (natural bbox) " << bb.desc() ; 

    nbbox cc = make_bbox( bb.min.x - 1 , bb.min.y - 1, bb.min.z - 1, bb.max.x + 1 , bb.max.y + 1, bb.max.z + 1 ) ;

    LOG(info) << " cc ( enlarged bbox ) " << cc.desc() ; 

    box->resizeToFit( cc , 1.f, 0.f );    

    box->pdump("after resizeToFit nbox::Create(2*h,2*h,2*h,0,CSG_BOX3)");

    nbbox cc2 = box->bbox_model();

    assert( cc.is_equal(cc2) );

    box->resizeToFit( cc , 1.f, 1.f );    
   
    nbbox cc3 = box->bbox_model();

    assert( !cc.is_equal(cc3) );
 
    LOG(info) << " cc3 ( enlarged bbox ) " << cc.desc() ; 
}



void test_resizeToFit_box()
{
    nbox* box = nbox::Create( 0.f, 0.f, -5.f, 10.f, CSG_BOX ); 
    box->verbosity = 3 ;  
    box->pdump("nbox::Create(0,0,-5,10, CSG_BOX)");


    float sz = 100.f ; 
    float dz = -50.f ; 
    nbbox bb = make_bbox( -sz, -sz, -sz + dz,   sz, sz, sz + dz ); 
    nbox* xbox = nbox::Create( 0.f, 0.f, dz, sz, CSG_BOX ); 

    LOG(info) << " bb " << bb.description() ; 

    float scale = 1.f ; 
    float delta = 0.f ; 
    box->resizeToFit( bb, scale , delta );  
    // ignores initial box, simply changes it to correspond to the bb 
    // BUT shifts of the bbox are honoured

    box->pdump("after resizeToFit "); 

    assert( box->is_equal(*xbox) ); 
}

void test_resizeToFit_box3()
{
    nbox* box = nbox::Create( 10.f, 10.f, 20.f, 0.f, CSG_BOX3 ); 
    box->verbosity = 3 ;  
    box->pdump("nbox::Create(10,10,20,0,CSG_BOX3)");


    float sz = 100.f ; 
    float dz = -50.f ; 
    nbbox bb = make_bbox( -sz, -sz, -sz + dz,   sz, sz, sz + dz ); 
    // this bb is shifted down in z 


    nbox* ybox = nbox::Create(2.f*sz, 2.f*sz, 2.f*sz, 0.f, CSG_BOX3 ); 

    LOG(info) << " bb " << bb.description() ; 

    float scale = 1.f ; 
    float delta = 0.f ; 
    box->resizeToFit( bb, scale , delta );  

    // ignores initial box, simply changes it to correspond to the bb 
    // also ignores any shifts in the bbox

    box->pdump("after resizeToFit "); 

    assert( box->is_equal(*ybox) ); 
}

void test_box()
{
    float sz = 10.f ; 
    float dz = -sz/2.f ;   // push down in z

    nbox* ybox = nbox::Create(0.f, 0.f, dz, sz, CSG_BOX ); 
    ybox->pdump("nbox::Create(0,0,-5, 10, CSG_BOX)");
    nbbox bb = ybox->bbox_model();
    LOG(info) << " bb " << bb.description() ; 

    nbbox xbb = make_bbox( -sz, -sz, -sz+dz  ,   sz, sz, sz+dz  ); 

    assert( bb.is_equal(xbb) );
}

void test_box3()   // always symmetric
{
    float sz = 10.f ; 

    nbox* box = nbox::Create( sz, sz, 2.f*sz, 0.f, CSG_BOX3 ); 
    box->pdump("nbox::Create(10,10,20, 0, CSG_BOX3)");
    nbbox bb = box->bbox_model();
    LOG(info) << " bb " << bb.description() ; 

    nbbox xbb = make_bbox( -sz/2.f, -sz/2.f, -sz  ,   sz/2.f, sz/2.f, sz  ); 
    assert( bb.is_equal(xbb) );
}


void test_box_transform()
{
    float sz = 10.f ; 
    nbox* a = nbox::Create(0.f, 0.f, 0.f, sz, CSG_BOX ); 
    a->pdump("nbox::Create(0,0,0, 10, CSG_BOX)");
 
    assert(  a->transform == NULL ); 
    a->transform = nmat4triple::make_translate( 0.f, 0.f, -5.f ); 
    a->update_gtransforms(); 

    nbbox bb = a->bbox();  // <-- global frame bbox, even for single primitive 

    LOG(info) << " bb " << bb.description() ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_resizeToFit();
    //test_resizeToFit_box();
    //test_resizeToFit_box3();

    //test_box();
    //test_box3();

    test_box_transform();

    return 0 ; 
}



