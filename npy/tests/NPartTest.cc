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

#include "NPart.hpp"
#include "PLOG.hh"

float unsigned_as_float(unsigned u)
{
  union {
    float f;
    unsigned u;
  } v1;

  v1.u = u;
  return v1.f;
}

unsigned float_as_unsigned(float f)
{
  union {
    float f;
    unsigned u;
  } v1;

  v1.f = f;
  return v1.u;
}


void test_p0()
{
    npart p ; 
    p.zero();
    p.dump("p0"); 
}


void test_p1()
{
    npart p ; 
    p.q0.f = { 0.f, 1.f, 2.f, 3.f };
    p.q3.u.w = 101 ; 
    p.q2.i.w = -101 ; 
    p.dump("p1"); 
}


/*
void test_NPART_TYPECODE()
{
    nvec4 buf[8] ; 

    buf[0] = {  0.f,  1.f,  2.f,  3.f } ;
    buf[1] = { 10.f, 11.f, 12.f, 13.f } ;
    buf[2] = { 20.f, 21.f, 22.f, 23.f } ;
    buf[3] = { 30.f, 31.f, 32.f, 33.f } ;

    buf[4] = {  0.f,  1.f,  2.f,  3.f } ;
    buf[5] = { 10.f, 11.f, 12.f, 13.f } ;
    buf[6] = { 20.f, 21.f, 22.f, 23.f } ;
    buf[7] = { 30.f, 31.f, 32.f, 33.f } ;
 

    float* bufPtr = (float*)&buf ; 

    unsigned tc0u = 100u ; 
    unsigned tc1u = 101u ; 
 
    float* tc0f = NPART_TYPECODE(NPART_OFFSET(bufPtr,0)) ;
    *tc0f = unsigned_as_float(tc0u) ;

    float* tc1f = NPART_TYPECODE(NPART_OFFSET(bufPtr,1)) ;
    *tc1f = unsigned_as_float(tc1u) ;

    unsigned tc0u_check = float_as_unsigned( *tc0f );
    unsigned tc1u_check = float_as_unsigned( *tc1f );

    std::cout << " tc0u_check : " << tc0u_check << std::endl ; 
    std::cout << " tc1u_check : " << tc1u_check << std::endl ; 

    assert(tc0u == tc0u_check);
    assert(tc1u == tc1u_check);
}
*/

void test_csgtree()
{
    npart* tree = new npart[3] ; 

    npart* root = tree + 0  ;
    npart* left = tree + 1 ;
    npart* right = tree + 2 ;

    left->zero();
    left->setTypeCode(CSG_SPHERE);
    left->setParam(0.f,0.f,50.f, 100.f) ;
    left->dump("left");


    right->zero();
    right->setTypeCode(CSG_SPHERE);
    right->setParam(0.f,0.f,-50.f, 100.f) ;
    right->dump("right");


    root->zero();
    root->setTypeCode(CSG_UNION);
    root->setLeft(1);     // 0-based array indices
    root->setRight(2);
    root->dump("right");


    npart::traverse( tree, 3, 0 ); 

}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_p0();
    test_p1();
 //   test_NPART_TYPECODE();

    test_csgtree();

  
    return 0 ; 
}




