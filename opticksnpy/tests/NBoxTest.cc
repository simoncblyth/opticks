#include "NGLMStream.hpp"

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

    bool verbose = false ; 
    glm::vec3 tlate ;

    for(int i=0 ; i < 100 ; i++)
    {
        gen(tlate); 

        glm::mat4 m = glm::translate(glm::mat4(1.0f), tlate );
        std::cout << " gtransform " << m << std::endl ; 

        nbox a = make_nbox(0.f,0.f,0.f,100.f);      
        // untouched box at origin

        nbox b = make_nbox(0.f,0.f,0.f,100.f);      
        b.gtransform = &m ; 
        // translated box via gtransform

        nbox c = make_nbox(-tlate.x,-tlate.y,-tlate.z,100.f);  
        // manually positioned box at negated tlate position 


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



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_gtransform();

    return 0 ; 
}




