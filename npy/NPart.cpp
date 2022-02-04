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

#include "SSys.hh"

#include "NPart.hpp"
#include "NBBox.hpp"
#include "NPlane.hpp"

#include "PLOG.hh"

#include <cstdio>
#include <cassert>

unsigned npart::VERSION = 1 ;   // could become an enumerated LAYOUT bitfield if need more granularity 

/*


        0   1   2   3 
       
    0   .   .   .   .

    1   .   .   .   .
  
    2   .   .   .   tc

    3   .   .   .   gt

*/


void npart::setTypeCode(OpticksCSG_t typecode)
{
    assert( TYPECODE_J == 2 && TYPECODE_K == 3 );
    q2.u.w = typecode ; 
}

void npart::setITransform(unsigned itransform_idx, bool complement)
{
    assert(VERSION == 1u);

    assert( TRANSFORM_J == 3 && TRANSFORM_K == 3 );

    unsigned ipack = itransform_idx & SSys::OTHERBIT32 ;
    if(complement) ipack |= SSys::SIGNBIT32 ; 

    LOG(debug) << "npart::setITransform"
             << " itransform_idx " << itransform_idx
             << " complement " << complement
             << " ipack " << ipack 
             << " ipack(hex) " << std::hex << ipack << std::dec 
             ; 

    q3.u.w = ipack ; 
}

void npart::setGTransform(unsigned gtransform_idx, bool complement)
{
    assert(VERSION == 1u);

   assert( GTRANSFORM_J == 3 && GTRANSFORM_K == 3 );

   unsigned gpack = gtransform_idx & SSys::OTHERBIT32 ;
   if(complement) gpack |= SSys::SIGNBIT32 ; 

   LOG(debug) << "npart::setGTransform"
             << " gtransform_idx " << gtransform_idx
             << " complement " << complement
             << " gpack " << gpack 
             << " gpack(hex) " << std::hex << gpack << std::dec 
             ; 

   q3.u.w = gpack ; 

}

// matches CSG/CSGNode
void npart::setSubNum(unsigned sub_num )
{
    q0.u.x = sub_num ; 
}
unsigned npart::getSubNum() const
{
    return q0.u.x ; 
}



// thought not used, but they are for in memory npart 
// the implicit left/rigth from the index is for the serialization

void npart::setLeft(unsigned left)
{
    qx.u.w = left ; 
}
void npart::setRight(unsigned right)
{
    qx.u.w = right ; 
}
unsigned npart::getLeft()
{
    return qx.u.w ; 
}
unsigned npart::getRight()
{
    return qx.u.w ; 
}







OpticksCSG_t npart::getTypeCode()
{
    return (OpticksCSG_t)q2.u.w ; 
}

bool npart::isPrimitive()
{
    return CSG::IsPrimitive(getTypeCode());
}

void npart::traverse( npart* tree, unsigned numNodes, unsigned i )
{
    assert( i < numNodes );

    npart* node = tree + i ; 
    bool prim = node->isPrimitive();
    printf("npart::traverse numNodes %u i %u prim %d \n", numNodes, i, prim) ;

    if(prim)
    {
        node->dump("traverse primitive");
    }
    else
    {
        unsigned l = node->getLeft();
        unsigned r = node->getRight();
        printf("npart::traverse (non-prim) numNodes %u i %u l %u r %u \n", numNodes, i, l, r) ;

        assert(l > 0 and r > 0);
         
        traverse( tree, numNodes, l ); 
        traverse( tree, numNodes, r ); 
    }
}



// primitive parts

void npart::dump(const char* msg) const 
{
    printf("%s\n", msg);
    q0.dump("q0");
    q1.dump("q1");
    q2.dump("q2");
    q3.dump("q3");

    qx.dump("qx");
}

void npart::zero()
{
    q0.u = {0,0,0,0} ;
    q1.u = {0,0,0,0} ;
    q2.u = {0,0,0,0} ;
    q3.u = {0,0,0,0} ;

    qx.u = {0,0,0,0} ;
}

void npart::check_bb_is_zero(OpticksCSG_t typecode) const 
{
   if( typecode == CSG_CONVEXPOLYHEDRON) return ;  // bbox is actually used 

   if( typecode == CSG_ZSPHERE )
   {
       if(q2.u.x != 3)
           LOG(fatal) << "check_bb_zero endcap flags expected 3 (ignored anyhow) " << q2.u.x ;
       //assert( q2.u.x == 3 );   // <-- no nolonger used endcap flags, but keeping it for matching 
        //TODO: check this is now zero 
   }
   else
   {
       assert( q2.u.x == 0 ); 

   } 
   assert( q2.u.y == 0 ); 
   assert( q2.u.z == 0 ); 

   assert( q3.u.x == 0 ); 
   assert( q3.u.y == 0 ); 
   assert( q3.u.z == 0 ); 
}


void npart::setParam(const nquad& q0_)
{
    assert( PARAM_J == 0 && PARAM_K == 0 );
    q0 = q0_;
}
void npart::setParam1(const nquad& q1_)
{
    assert( PARAM1_J == 1 && PARAM_K == 0 );
    q1 = q1_;
}
void npart::setParam2(const nquad& q2_)
{
    q2 = q2_;
}
void npart::setParam3(const nquad& q3_)
{
    q3 = q3_;
}




void npart::setParam(float x, float y, float z, float w)
{
    nquad param ;
    param.f = {x,y,z,w} ;
    setParam( param );
}
void npart::setParam1(float x, float y, float z, float w)
{
    nquad param1 ;
    param1.f = {x,y,z,w} ;
    setParam1( param1 );
}






void npart::setBBox(const nbbox& bb)
{
    assert(VERSION == 0u || VERSION == 1u);
    // used by CSG_ZLENS GMaker::makeZSphereIntersect
  

    assert( BBMIN_J == 2 && BBMIN_K == 0 );
    q2.f.x = bb.min.x ; 
    q2.f.y = bb.min.y ; 
    q2.f.z = bb.min.z ;
 
    assert( BBMAX_J == 3 && BBMAX_K == 0 );
    q3.f.x = bb.max.x ; 
    q3.f.y = bb.max.y ; 
    q3.f.z = bb.max.z ;
}

