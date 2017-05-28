#include <cstdlib>
#include <cfloat>
#include "NGLMExt.hpp"


#include "NGenerator.hpp"
#include "NBox.hpp"
#include "NBBox.hpp"

#include "PLOG.hh"


void test_gtransform()
{
    nbbox bb ; 
    bb.min = {-200.f, -200.f, -200.f };
    bb.max = { 200.f,  200.f,  200.f };

    NGenerator gen(bb);

    bool verbose = !!getenv("VERBOSE")  ; 
    glm::vec3 tlate ;

    for(int i=0 ; i < 100 ; i++)
    {
        gen(tlate); 

        glm::mat4 t = glm::translate(glm::mat4(1.0f), tlate );
        glm::mat4 v = nglmext::invert_tr(t);
        glm::mat4 q = glm::transpose(v); 

        //nmat4pair pair(tr, irit);
        nmat4triple triple(t, v, q);

        if(verbose)
        std::cout << " gtransform " << triple << std::endl ; 

        nbox a = make_box(0.f,0.f,0.f,100.f);      
        // untouched box at origin

        nbox b = make_box(0.f,0.f,0.f,100.f);      
        b.gtransform = &triple ;  
        // translated box via gtransform

        nbox c = make_box( tlate.x, tlate.y, tlate.z,100.f);  
        // manually positioned box at tlated position 


        float x = 0 ; 
        float y = 0 ; 
        float z = 0 ; 

        for(int iz=-200 ; iz <= 200 ; iz+= 10 ) 
        {
           z = iz ;  
           float a_ = a(x,y,z) ;
           float b_ = b(x,y,z) ;
           float c_ = c(x,y,z) ;
      
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


void test_sdf()
{
    nbox b = make_box(0,0,0,1);  // unit box centered at origin       

    for(float x=-2.f ; x < 2.f ; x+=0.01f )
    {
        float sd1 = b.sdf1(x,0,0) ;
        float sd2 = b.sdf2(x,0,0) ;

       /*
        std::cout
            << " x " << std::setw(5) << x 
            << " sd1 " << std::setw(5) << sd1 
            << " sd2 " << std::setw(5) << sd2
            << std::endl ;  
        */

        assert(sd1 == sd2 );
    }

}


void test_parametric()
{
    LOG(info) << "test_parametric" ;

    nbox box = make_box(0,0,0,100); 

    unsigned nsurf = box.par_nsurf();
    assert(nsurf == 6);

    unsigned nu = 1 ; 
    unsigned nv = 1 ; 

    for(unsigned surf=0 ; surf < nsurf ; surf++)
    {
        std::cout << " surf : " << surf << std::endl ; 

        for(unsigned u=0 ; u <= nu ; u++){
        for(unsigned v=0 ; v <= nv ; v++)
        {
            nquad quv ;
            quv.i.x = u ; 
            quv.i.y = v ; 
            quv.i.z = nu ; 
            quv.i.w = nv ; 

            glm::vec3 p = box.par_pos(quv, surf );

            std::cout 
                 << " u " << std::setw(3) << u  
                 << " v " << std::setw(3) << v
                 << " p " << glm::to_string(p)
                 << std::endl ;   
        }
        }
    }
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    //test_gtransform();
    //test_sdf();
    test_parametric();

    return 0 ; 
}




