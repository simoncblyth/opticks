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


#include "NGLMExt.hpp"
#include "nmat4pair.hpp"
#include "nmat4triple.hpp"
#include "GLMFormat.hpp"

#include "NNode.hpp"
#include "NNodePoints.hpp"
#include "NSphere.hpp"
#include "NCylinder.hpp"
#include "NCone.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"
#include "NBBox.hpp"
#include "NBox.hpp"
#include "Nuv.hpp"

#include "OPTICKS_LOG.hh"


nmat4pair make_tlate_pair( const glm::vec3& tlate )
{
    glm::mat4 tr(1.f) ;
    tr = glm::translate( tr, tlate );

    glm::mat4 irit = nglmext::invert_tr( tr );

    return nmat4pair(tr, irit);
}


nmat4triple make_tlate_triple( const glm::vec3& tlate )
{
    glm::mat4 tr(1.f) ;
    tr = glm::translate( tr, tlate );

    glm::mat4 irit = nglmext::invert_tr( tr );
    glm::mat4 irit_T = glm::transpose(irit);

    return nmat4triple(tr, irit, irit_T);
}


void test_node_transforms()
{
    LOG(info) << "test_node_transforms" ; 

    nmat4triple mpx = make_tlate_triple( glm::vec3(300,0,0) );
    nmat4triple mpy = make_tlate_triple( glm::vec3(0,300,0) );
    nmat4triple mpz = make_tlate_triple( glm::vec3(0,0,300) );

    std::cout << " mpx  " << mpx << std::endl ; 
    std::cout << " mpy  " << mpy << std::endl ; 
    std::cout << " mpz  " << mpz << std::endl ; 

    /*


                   u
                   

            lu            ru

        la     lb     ra      rb


    */
  
    // lu
    nsphere* la = make_sphere(-500.f,0.f,-50.f,100.f); la->label = "la" ; 
    nsphere* lb = make_sphere(-500.f,0.f, 50.f,100.f); lb->label = "lb" ; 
    nunion*  lu = nunion::make_union( la, lb );
    la->parent = lu ; 
    lb->parent = lu ; 

    // ru
    nsphere* ra = make_sphere( 500.f,0.f,-50.f,100.f); ra->label = "ra" ; 
    nsphere* rb = make_sphere( 500.f,0.f, 50.f,100.f); rb->label = "rb" ; 
    nunion*  ru = nunion::make_union( ra, rb );
    ra->parent = ru ; 
    rb->parent = ru ; 

    // u 
    nunion* u = nunion::make_union( lu, ru );
    lu->parent = u ; 
    ru->parent = u ; 
 

    u->transform = &mpx ; 
    ru->transform = &mpy ; 
    rb->transform = &mpz ;     
     

    // setting gtransform on internal nodes does nothing ... need to do that to the leaves
    u->update_gtransforms();

    assert(ra->gtransform);
    assert(rb->gtransform);
    assert(la->gtransform);
    assert(lb->gtransform);

    std::cout << " rb->gt " << *(rb->gtransform) << std::endl ; 

    std::vector<glm::vec3> centers ; 
    std::vector<glm::vec3> dirs ; 
    u->collect_prim_centers(centers, dirs);

    unsigned ncen = centers.size();
    unsigned ndir = dirs.size();
    assert( ncen == ndir );

    for(unsigned i=0 ; i < ncen ; i++) std::cout << i << " center:" << centers[i] << " dir:" << dirs[i] << std::endl ; 

    u->verbosity = 1 ;  
    u->dump("dump(v:1)"); 

    std::cout << std::endl ; 
    u->verbosity = 2 ;  
    u->dump("dump(v:2)"); 

    // hmm need to do this with rotations, as with translation order doesnt matter 
}

void test_node_bbox_unspsp()
{
    nsphere* a = make_sphere(-50.f,0.f,0.f,100.f); a->label = "a" ; 
    nsphere* b = make_sphere( 50.f,0.f,0.f,100.f); b->label = "b" ; 
    nunion*  ab = nunion::make_union( a, b );
    a->parent = ab ; 
    b->parent = ab ; 

    nbbox bb = ab->bbox(); 

    LOG(info) << bb.desc();  
}


void test_node_bbox_uncyco()
{
    float cy_radius = 100.f ; 
    float cy_z1 = -50.f ; 
    float cy_z2 =  50.f ; 

    ncylinder* cy = make_cylinder(cy_radius,cy_z1,cy_z2); 
    cy->label = "cy" ; 

    float co_r1 = 100.f ; 
    float co_r2 =  10.f ; 
    float co_z1 =  50.f ; 
    float co_z2 = 100.f ; 
    
    ncone* co = make_cone(co_r1,co_z1,co_r2, co_z2); 
    co->label = "co" ; 

    nunion*  uncyco = nunion::make_union( cy, co );
    cy->parent = uncyco ; 
    co->parent = uncyco ; 

    nbbox bb = uncyco->bbox(); 

    LOG(info) << bb.desc();  
}






void test_getSurfacePoints_difference()
{
    LOG(info) << "test_getSurfacePoints_difference" ; 

    nbox* a = make_box3(400,400,100);
    nbox* b = make_box3(300,300,50);

    glm::vec3 tlate(0,0,25);
    b->transform = nmat4triple::make_translate( tlate );    

    ndifference* obj = ndifference::make_difference(a, b);

    a->parent = obj ;  // parent hookup usually done by NCSG::import_r 
    b->parent = obj ;   
    
    obj->update_gtransforms();  // recurse over tree using parent links to set gtransforms
    // without update_gtransforms they are all NULL so the dump shows un-translated bbox

    obj->dump("before scaling obj");


    glm::vec3 tsca(2,2,2);
    obj->transform = nmat4triple::make_scale( tsca );    

    obj->update_gtransforms();  
   
    obj->dump("after scaling obj");

    // NOW HOW TO GET LOCALS ? WITHOUT THE OBJ TRANSFORM  

    b->verbosity = 3 ; 
    b->pdump("b.pdump");


}


void test_getSurfacePoints()
{
     nbox* bx = make_box3(20,20,20);

     unsigned ns = bx->par_nsurf();
     assert( ns == 6 );

     unsigned level = 1 ;
     int margin = 1 ; 

     /*

     level 1  
          (1 << 1) = 2 divisions in u and v,  +1 for endpost -> 3 points in u and v 

     margin 1
          skip the start and end of uv range 
          3 points - 2*margin ->   1 point   at mid uv of the face 
     

          +---+---+
          |   |   |
          +---*---+
          |   |   |
          +---+---+

     */

     int ndiv = (1 << level) + 1 - 2*margin ;
     unsigned expect = ndiv*ndiv  ;  
     unsigned verbosity = 1 ; 
     unsigned prim_idx = 0 ; 

     LOG(info) << "test_getSurfacePoints" 
               << " level " << level 
               << " margin " << margin 
               << " ((1 << level) + 1) =  " << (1 << level) + 1
               << " (((1 << level) + 1) - 2*margin) =  " << ((1 << level) + 1) - 2*margin
               << " ndiv " << ndiv
               << " expect " << expect 
                ; 

     bx->collectParPoints(prim_idx, level, margin, FRAME_LOCAL, verbosity) ;
     const std::vector<glm::vec3>& pts = bx->par_points ; 

     assert( pts.size() == expect*ns );

     for(unsigned i=0 ; i < pts.size() ; i++ ) 
          std::cout 
              << " pts " << pts.size() 
              << " i " << i
              << " " 
              << glm::to_string(pts[i]) << std::endl ; 
}




void test_getCoincidentSurfacePoints()
{
     LOG(info) << "test_getCoincidentSurfacePoints" ; 
/*


         
         +       +
         
             0          

         +       +
       -10       10

  
*/
     nbox* bx = make_box(0,0,0,10);
     nbox* other = make_box(0,0,20,10) ;  

     unsigned ns = bx->par_nsurf();
     assert( ns == 6 );

    
    // expect coincident +Z face of bx with other

     float epsilon = 1e-5 ; 
     unsigned level  = 1 ;   //  (1 << level) divisions in u and v, level 0 -> u [0,1] inclusive, level 1 [0,1,2,3,4]  
     int margin = 1 ;   //  when >0 skips uv edges/corners 

     for(unsigned s = 0 ; s < ns ; s++)
     {    
         std::vector<nuv> coincident ; 

         bx->getCoincidentSurfacePoints(coincident, s, level, margin, other, epsilon, FRAME_LOCAL ) ;

         std::cout << " s " 
                   << s 
                   << " coincident " << coincident.size()
                   << std::endl ; 

         for(unsigned i=0 ; i < coincident.size() ; i++ ) std::cout << coincident[i].desc() << std::endl ; 
     }
}



void test_getCoincident()
{
     nbox* bx = make_box(0,0,0,10);
     nbox* other = make_box(20,0,0,10) ;  

     std::vector<nuv> coincident ; 
     bx->getCoincident(coincident, other ); 

     LOG(info) << "test_getCoincident" 
               << " coincident " << coincident.size()
               ;

     for(unsigned i=0 ; i < coincident.size() ; i++ ) 
     {
         nuv uv = coincident[i] ;

         glm::vec3 pos = bx->par_pos_global(uv);

         std::cout 
              <<  " uv " << uv.desc() 
              <<  " pos " << glm::to_string(pos)
              << std::endl 
              ; 

     }
 
}



void test_getSurfacePointsAll_Composite()
{
    LOG(info) << "test_getSurfacePointsAll_Composite" ; 

    glm::vec3 tlate(10,0,10);
    glm::vec4 aa(0,0,0,10);
    glm::vec4 bb(0,0,0,10);



    //glm::vec3 scale(1000,1000,1000);
    //glm::mat4 sc = nglmext::make_scale( scale ); 
    //const glm::mat4* ndtr = &sc ;  // <-- mock structural transform 


    {
        nbox* a = make_box(aa.x,aa.y,aa.z,aa.w);
        nbox* b = make_box(bb.x,bb.y,bb.z,bb.w);
        b->transform = nmat4triple::make_translate( tlate );    

        ndifference* ab = ndifference::make_difference(a, b); 

        a->parent = ab ;  // parent hookup usually done by NCSG::import_r 
        b->parent = ab ;   
        ab->update_gtransforms();  // recurse over tree using parent links to set gtransforms

        ab->dump();

        NNodePoints pts(ab, NULL );
        glm::uvec4 tot = pts.collect_surface_points();
        pts.dump();
        std::cout << "difference:(inside/surface/outside/select)  " << glm::to_string(tot) << std::endl ; 
    }
    {
        nbox* a = make_box(aa.x,aa.y,aa.z,aa.w);
        nbox* b = make_box(bb.x,bb.y,bb.z,bb.w);
        b->transform = nmat4triple::make_translate( tlate );    

        nunion*        ab = nunion::make_union(a, b); 

        a->parent = ab ;  // parent hookup usually done by NCSG::import_r 
        b->parent = ab ;   
        ab->update_gtransforms();  // recurse over tree using parent links to set gtransforms

        ab->dump();

        NNodePoints pts(ab, NULL );
        glm::uvec4 tot = pts.collect_surface_points();
        pts.dump();
        std::cout << "union: (inside/surface/outside/select)   " << glm::to_string(tot) << std::endl ; 
    }
    {
        nbox* a = make_box(aa.x,aa.y,aa.z,aa.w);
        nbox* b = make_box(bb.x,bb.y,bb.z,bb.w);
        b->transform = nmat4triple::make_translate( tlate );    

        nintersection* ab = nintersection::make_intersection(a, b); 

        a->parent = ab ;  // parent hookup usually done by NCSG::import_r 
        b->parent = ab ;   
        ab->update_gtransforms();  // recurse over tree using parent links to set gtransforms

        ab->dump();

        NNodePoints pts(ab, NULL );
        glm::uvec4 tot = pts.collect_surface_points();
        pts.dump();
        std::cout << "intersection:  (inside/surface/outside/select)  " << glm::to_string(tot) << std::endl ; 
    } 
}






int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_node_transforms();
    //test_node_bbox_unspsp();
    test_node_bbox_uncyco();

    //test_getSurfacePoints_difference();
    //test_getSurfacePoints();
    //test_getCoincidentSurfacePoints();
    //test_getCoincident();

    //test_getSurfacePointsAll_Composite();

    return 0 ; 
}



