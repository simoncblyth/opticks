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
#include "NNodeSample.hpp"
#include "NNode.hpp"

#include "PLOG.hh"
#include "NPrimitives.hpp"


nnode* NNodeSample::Sphere1()
{
    nsphere* a = nsphere::Create(0.f,0.f,-50.f,100.f);
    return a ; 
}
nnode* NNodeSample::Sphere2()
{
    nsphere* b = nsphere::Create(0.f,0.f, 50.f,100.f);
    return b ; 
}
nnode* NNodeSample::Union1()
{
    nnode* a = Sphere1() ; 
    nnode* b = Sphere2() ; 
    nunion* u = nunion::make_union( a, b );
    return u ; 
}
nnode* NNodeSample::Intersection1()
{
    nnode* a = Sphere1() ; 
    nnode* b = Sphere2() ; 
    nintersection* i = nintersection::make_intersection( a, b ); 
    return i ; 
}
nnode* NNodeSample::Difference1()
{
    nnode* a = Sphere1() ; 
    nnode* b = Sphere2() ; 
    ndifference* d1 = ndifference::make_difference( a, b ); 
    return d1 ; 
}
nnode* NNodeSample::Difference2()
{
    nnode* a = Sphere1() ; 
    nnode* b = Sphere2() ; 
    ndifference* d2 = ndifference::make_difference( b, a ); 
    return d2 ; 
}
nnode* NNodeSample::Union2()
{
    nnode* d1 = Difference1() ; 
    nnode* d2 = Difference2() ; 
    nunion* u2 = nunion::make_union( d1, d2 );
    return u2 ; 
}
nnode* NNodeSample::Box()
{
    nbox*    c = nbox::Create(0.f,0.f,0.f,200.f, CSG_BOX);
    return c ; 
}
nnode* NNodeSample::SphereBoxUnion()
{
    float radius = 200.f ; 
    float inscribe = 1.3f*radius/sqrt(3.f) ; 

    nsphere* sp = nsphere::Create(0.f,0.f,0.f,radius );
    nbox*    bx = nbox::Create(0.f,0.f,0.f, inscribe, CSG_BOX);
    nunion*  u_sp_bx = nunion::make_union( sp, bx );

    return u_sp_bx ;
}
nnode* NNodeSample::SphereBoxIntersection()
{
    float radius = 200.f ; 
    float inscribe = 1.3f*radius/sqrt(3.f) ; 

    nsphere* sp = nsphere::Create(0.f,0.f,0.f,radius );
    nbox*    bx = nbox::Create(0.f,0.f,0.f, inscribe, CSG_BOX );
    nintersection*  i_sp_bx = nintersection::make_intersection( sp, bx );

    return i_sp_bx ;
}
nnode* NNodeSample::SphereBoxDifference()
{
    float radius = 200.f ; 
    float inscribe = 1.3f*radius/sqrt(3.f) ; 

    nsphere* sp = nsphere::Create(0.f,0.f,0.f,radius );
    nbox*    bx = nbox::Create(0.f,0.f,0.f, inscribe, CSG_BOX );
    ndifference*    d_sp_bx = ndifference::make_difference( sp, bx );

    return d_sp_bx ;
}
nnode* NNodeSample::BoxSphereDifference()
{
    float radius = 200.f ; 
    float inscribe = 1.3f*radius/sqrt(3.f) ; 

    nsphere* sp = nsphere::Create(0.f,0.f,0.f,radius );
    nbox*    bx = nbox::Create(0.f,0.f,0.f, inscribe, CSG_BOX );
    ndifference*    d_bx_sp = ndifference::make_difference( bx, sp );

    return d_bx_sp ;
}

void NNodeSample::Tests(std::vector<nnode*>& nodes )
{
    nodes.push_back( Sphere1() ) ; 
    nodes.push_back( Sphere2() ) ; 
    nodes.push_back( Union1() ) ; 
    nodes.push_back( Intersection1() ) ; 
    nodes.push_back( Difference1() ) ; 
    nodes.push_back( Difference2() ) ; 
    nodes.push_back( Union2() ) ; 
    nodes.push_back( Box() ) ; 
    nodes.push_back( SphereBoxUnion() ) ; 
    nodes.push_back( SphereBoxIntersection() ) ; 
    nodes.push_back( SphereBoxDifference() ) ; 
    nodes.push_back( BoxSphereDifference() ) ; 
}


nnode* NNodeSample::_Prepare(nnode* root)
{
    root->update_gtransforms();
    root->verbosity = SSys::getenvint("VERBOSITY", 1) ; 
    root->dump() ; 
    const char* boundary = "Rock//perfectAbsorbSurface/Vacuum" ;
    root->set_boundary(boundary); 
    return root ; 
}
nnode* NNodeSample::DifferenceOfSpheres()
{
    nsphere* a = nsphere::Create( 0.000,0.000,0.000,500.000 ) ; a->label = "a" ;   
    nsphere* b = nsphere::Create( 0.000,0.000,0.000,100.000 ) ; b->label = "b" ;   
    ndifference* ab = ndifference::make_difference( a, b ) ; ab->label = "ab" ; a->parent = ab ; b->parent = ab ;  ;   
    return _Prepare(ab) ; 
}
nnode* NNodeSample::Box3()
{
    nbox* a = nbox::Create(300.f,300.f,200.f,0.f, CSG_BOX3) ; 
    return _Prepare(a) ; 
}

/**
NNodeSample::Cone_0
---------------------


                             (2,2)
                             (r2,z2)
                  +-------------+
                 /               \
                /                 \
               /                   \
              /                     \
             /                       \
            /                         \
           +-------------0-------------+ 

                                      (r1, z1) 
                                      (4, 0) 
**/

nnode* NNodeSample::Cone_0(const char* name)
{
    LOG(info) << name ; 
    float r1 = 4.f ; 
    float z1 = 0.f ;
    float r2 = 2.f ; 
    float z2 = 2.f ;
    ncone* cone = make_cone(r1,z1,r2,z2) ; 
    nnode* node = (nnode*)cone ;
    return node ; 
}






nnode* NNodeSample::Sample(const char* name)
{
    if(strcmp(name, "DifferenceOfSpheres") == 0) return DifferenceOfSpheres();
    if(strcmp(name, "Box3") == 0)                return Box3();
    if(strcmp(name, "Cone_0") == 0)              return Cone_0(name);
    return NULL ; 
}


