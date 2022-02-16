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

// TEST=NSphereTest om-t

#include <cstdlib>
#include "NGLMExt.hpp"
#include "nmat4triple.hpp"

#include "NGenerator.hpp"
#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"
#include "NBBox.hpp"

#include "OPTICKS_LOG.hh"



void test_dumpSurfacePointsAll()
{
    LOG(info) << "test_dumpSurfacePointsAll" ;
    nsphere* sp = nsphere::Create(100.f);
    sp->dumpSurfacePointsAll("sp.dumpSurfacePointsAll", FRAME_LOCAL);
}


void test_part()
{
    nsphere* s = nsphere::Create(0,0,3,10);
    npart p = s->part();
    p.dump("p");
}

void test_intersect()
{
    nsphere* s1 = nsphere::Create(0,0,3,10);
    nsphere* s2 = nsphere::Create(0,0,1,10);

    ndisk* d12 = nsphere::intersect(s1,s2) ;
    d12->dump("d12");


    npart s1l = s1->zlhs(d12);
    s1l.dump("s1l");

    npart s1r = s1->zrhs(d12);
    s1r.dump("s1r");
}



void test_sdf()
{
    LOG(info) << "test_sdf" ; 

    nsphere* a = nsphere::Create(0.f,0.f,-50.f,100.f);
    nsphere* b = nsphere::Create(0.f,0.f,-50.f,100.f);
    b->complement = true ; 

    float x = 0.f ; 
    float y = 0.f ; 
    float z = 0.f ; 

    float epsilon = 1e-5 ; 

    for(int iz=-200 ; iz <= 200 ; iz+= 10, z=iz ) 
    {
        float sd_a = (*a)(x,y,z) ;
        float sd_b = (*b)(x,y,z) ;

        assert( abs( sd_a + sd_b) < epsilon );

        std::cout 
             << " z " << std::setw(10) << z 
             << " sd_a  " << std::setw(10) << std::fixed << std::setprecision(2) << sd_a
             << " sd_b " << std::setw(10) << std::fixed << std::setprecision(2) << sd_b
             << std::endl 
             ; 
    }
}

void test_diff_DeMorgan_sdf()
{
    LOG(info) << "test_diff_DeMorgan_sdf" ; 

    nsphere* a = nsphere::Create(0.f,0.f,-50.f,100.f);
    nsphere* b = nsphere::Create(0.f,0.f, 50.f,100.f);
    nsphere* c = nsphere::Create(0.f,0.f, 50.f,100.f);
    c->complement = true ; 

    ndifference*   d = ndifference::make_difference( a, b ); 
    nintersection* i = nintersection::make_intersection( a, c ); 

    float epsilon = 1e-5 ; 
    float x = 0.f ; 
    float y = 0.f ; 
    float z = 0.f ; 

    for(int iz=-200 ; iz <= 200 ; iz+= 10, z=iz ) 
    {
        float sd_d = (*d)(x,y,z) ;
        float sd_i = (*i)(x,y,z) ;


        std::cout 
             << " z " << std::setw(10) << z 
             << " sd_d  " << std::setw(10) << std::fixed << std::setprecision(2) << sd_d
             << " sd_i " << std::setw(10) << std::fixed << std::setprecision(2) << sd_i
             << std::endl 
             ; 

        assert( abs( sd_d - sd_i) < epsilon );
    }
}



void test_csgsdf()
{
    nsphere* a = nsphere::Create(0.f,0.f,-50.f,100.f);
    nsphere* b = nsphere::Create(0.f,0.f, 50.f,100.f);

    nunion* u = nunion::make_union( a, b );
    nintersection* i = nintersection::make_intersection( a, b ); 
    ndifference* d1 = ndifference::make_difference( a, b ); 
    ndifference* d2 = ndifference::make_difference( b, a ); 
    nunion* u2 = nunion::make_union( d1, d2 );

    typedef std::vector<nnode*> VN ;

    VN nodes ; 
    nodes.push_back( (nnode*)a );
    nodes.push_back( (nnode*)b );
    nodes.push_back( (nnode*)u );
    nodes.push_back( (nnode*)i );
    nodes.push_back( (nnode*)d1 );
    nodes.push_back( (nnode*)d2 );
    nodes.push_back( (nnode*)u2 );

    for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
    {
        nnode* n = *it ; 
        OpticksCSG_t type = n->type ; 
        const char* name = n->csgname();
        std::cout 
                  << " type: " << std::setw(3) << type 
                  << " name: " << ( name ? name : "-" ) 
                  << " sdf(0,0,0): " << std::setw(10) << std::fixed << std::setprecision(2) << (*n)(0,0,0)
                  << std::endl 
                  ; 

    }

    float x = 0.f ; 
    float y = 0.f ; 
    float z = 0.f ; 

    for(int iz=-200 ; iz <= 200 ; iz+= 10, z=iz ) 
    {
        std::cout 
             << " z  " << std::setw(10) << z 
             << " a  " << std::setw(10) << std::fixed << std::setprecision(2) << (*a)(x,y,z) 
             << " b  " << std::setw(10) << std::fixed << std::setprecision(2) << (*b)(x,y,z) 
             << " u  " << std::setw(10) << std::fixed << std::setprecision(2) << (*u)(x,y,z) 
             << " i  " << std::setw(10) << std::fixed << std::setprecision(2) << (*i)(x,y,z) 
             << " d1 " << std::setw(10) << std::fixed << std::setprecision(2) << (*d1)(x,y,z) 
             << " d2 " << std::setw(10) << std::fixed << std::setprecision(2) << (*d2)(x,y,z) 
             << " u2 " << std::setw(10) << std::fixed << std::setprecision(2) << (*u2)(x,y,z) 
             << std::endl 
             ; 

        std::cout << " z  " << std::setw(10) << z  ;
        for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
        {
             nnode* n = *it ; 
             const char* name = n->csgname();
             printf(" %.3s %10.2f", ( name ? name : "-" ), (*n)(x,y,z) ); 
        }
        std::cout << std::endl ; 
    }
}


void test_bbox()
{
    nsphere* a = nsphere::Create(0.f,0.f,-50.f,100.f);
    a->dump("sph");

    nbbox bb = a->bbox();
    bb.dump("bb");
}

void test_bbox_u()
{
    nsphere* a = nsphere::Create(0.f,0.f,-50.f,100.f);
    nsphere* b = nsphere::Create(0.f,0.f, 50.f,100.f);
    nunion*  u = nunion::make_union( a, b );

    a->dump("(a) sph");
    b->dump("(b) sph");
    u->dump("(u) union(a,b)");

    nbbox a_bb = a->bbox();
    a_bb.dump("(a) bb");

    nbbox b_bb = b->bbox();
    b_bb.dump("(b) bb");

    nbbox u_bb = u->bbox();
    u_bb.dump("(u) bb");
}


void test_gtransform()
{
    nbbox bb ; 
    bb.min = {-200.f, -200.f, -200.f };
    bb.max = { 200.f,  200.f,  200.f };

    NGenerator gen(bb);

    bool verbose = !!getenv("VERBOSE") ; 
    glm::vec3 tlate ;

    for(int i=0 ; i < 100 ; i++)
    {
        gen(tlate); 

        glm::mat4 tr = glm::translate(glm::mat4(1.0f), tlate );
        glm::mat4 irit = nglmext::invert_tr(tr);
        glm::mat4 irit_T = glm::transpose(irit) ;

        //nmat4pair mp(tr, irit);
        nmat4triple triple(tr, irit, irit_T);

        if(verbose)
        std::cout << " gtransform " << triple << std::endl ; 

        nsphere* a = nsphere::Create(0.f,0.f,0.f,100.f);      
        // untouched sphere at origin

        nsphere* b = nsphere::Create(0.f,0.f,0.f,100.f);      
        b->gtransform = &triple ; 
        // translated sphere via gtransform

        nsphere* c = nsphere::Create( tlate.x, tlate.y, tlate.z,100.f);  
        // manually positioned sphere at tlate-d position 


        float x = 0 ; 
        float y = 0 ; 
        float z = 0 ; 

        for(int iz=-200 ; iz <= 200 ; iz+= 10 ) 
        {
           z = iz ;  
           float a_ = (*a)(x,y,z) ;
           float b_ = (*b)(x,y,z) ;
           float c_ = (*c)(x,y,z) ;
      
           if(verbose) 
           std::cout 
                 << " z " << std::setw(10) << z 
                 << " a_ " << std::setw(10) << std::fixed << std::setprecision(2) << a_
                 << " b_ " << std::setw(10) << std::fixed << std::setprecision(2) << b_
                 << " c_ " << std::setw(10) << std::fixed << std::setprecision(2) << c_
                 << std::endl 
                 ; 

           assert( b_ == c_ );

        }
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_part();
    test_intersect();

    test_csgsdf();

    test_bbox();
    test_bbox_u();
    test_gtransform();

    test_sdf();
    test_diff_DeMorgan_sdf();
    test_dumpSurfacePointsAll();

    return 0 ; 
}




