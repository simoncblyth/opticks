
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NNode.hpp"
#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"
#include "NBBox.hpp"
#include "NBox.hpp"
#include "Nuv.hpp"

#include "PLOG.hh"


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
    nsphere la = make_sphere(-500.f,0.f,-50.f,100.f); la.label = "la" ; 
    nsphere lb = make_sphere(-500.f,0.f, 50.f,100.f); lb.label = "lb" ; 
    nunion  lu = make_union( &la, &lb );
    la.parent = &lu ; 
    lb.parent = &lu ; 

    // ru
    nsphere ra = make_sphere( 500.f,0.f,-50.f,100.f); ra.label = "ra" ; 
    nsphere rb = make_sphere( 500.f,0.f, 50.f,100.f); rb.label = "rb" ; 
    nunion  ru = make_union( &ra, &rb );
    ra.parent = &ru ; 
    rb.parent = &ru ; 

    // u 
    nunion u = make_union( &lu, &ru );
    lu.parent = &u ; 
    ru.parent = &u ; 
 

    u.transform = &mpx ; 
    ru.transform = &mpy ; 
    rb.transform = &mpz ;     
     

    // setting gtransform on internal nodes does nothing ... need to do that to the leaves
    u.update_gtransforms();

    assert(ra.gtransform);
    assert(rb.gtransform);
    assert(la.gtransform);
    assert(lb.gtransform);

    std::cout << " rb.gt " << *rb.gtransform << std::endl ; 

    std::vector<glm::vec3> centers ; 
    std::vector<glm::vec3> dirs ; 
    u.collect_prim_centers(centers, dirs);

    unsigned ncen = centers.size();
    unsigned ndir = dirs.size();
    assert( ncen == ndir );

    for(unsigned i=0 ; i < ncen ; i++) std::cout << i << " center:" << centers[i] << " dir:" << dirs[i] << std::endl ; 

    u.verbosity = 1 ;  
    u.dump_prim("prim(v:1)"); 

    std::cout << std::endl ; 
    u.verbosity = 2 ;  
    u.dump_prim("prim(v:2)"); 

    // hmm need to do this with rotations, as with translation order doesnt matter 
}



void test_getSurfacePoints()
{

     nbox bx = make_box(0,0,0,10);

     unsigned ns = bx.par_nsurf();
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

     LOG(info) << "test_getSurfacePoints" 
               << " level " << level 
               << " margin " << margin 
               << " ((1 << level) + 1) =  " << (1 << level) + 1
               << " (((1 << level) + 1) - 2*margin) =  " << ((1 << level) + 1) - 2*margin
               << " ndiv " << ndiv
               << " expect " << expect 
                ; 

     for(unsigned s = 0 ; s < ns ; s++)
     {    
         std::vector<glm::vec3> surf ; 

         bx.getSurfacePoints(surf, s, level, margin) ;

         //assert( surf.size() == expect );

         std::cout 
              << " s " << s
              << " surf " << surf.size() 
              << std::endl ; 

         for(unsigned i=0 ; i < surf.size() ; i++ ) std::cout << glm::to_string(surf[i]) << std::endl ; 
     }
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
     nbox bx = make_box(0,0,0,10);
     nbox other = make_box(0,0,20,10) ;  

     unsigned ns = bx.par_nsurf();
     assert( ns == 6 );

    
    // expect coincident +Z face of bx with other

     float epsilon = 1e-5 ; 
     unsigned level  = 1 ;   //  (1 << level) divisions in u and v, level 0 -> u [0,1] inclusive, level 1 [0,1,2,3,4]  
     int margin = 1 ;   //  when >0 skips uv edges/corners 

     for(unsigned s = 0 ; s < ns ; s++)
     {    
         std::vector<nuv> coincident ; 

         bx.getCoincidentSurfacePoints(coincident, s, level, margin, &other, epsilon ) ;

         std::cout << " s " 
                   << s 
                   << " coincident " << coincident.size()
                   << std::endl ; 

         for(unsigned i=0 ; i < coincident.size() ; i++ ) std::cout << coincident[i].desc() << std::endl ; 
     }
}



void test_getCoincident()
{
     nbox bx = make_box(0,0,0,10);
     nbox other = make_box(20,0,0,10) ;  

     std::vector<nuv> coincident ; 
     bx.getCoincident(coincident, &other ); 

     LOG(info) << "test_getCoincident" 
               << " coincident " << coincident.size()
               ;

     for(unsigned i=0 ; i < coincident.size() ; i++ ) 
     {
         nuv uv = coincident[i] ;

         glm::vec3 pos = bx.par_pos(uv);

         std::cout 
              <<  " uv " << uv.desc() 
              <<  " pos " << glm::to_string(pos)
              << std::endl 
              ; 

     }
 
}







int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    //test_node_transforms();

    test_getSurfacePoints();
    test_getCoincidentSurfacePoints();
    test_getCoincident();

    return 0 ; 
}



