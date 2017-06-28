#include <iostream>
#include <iomanip>

#include "PLOG.hh"
#include "Nuv.hpp"
#include "NDisc.hpp"



void test_parametric()
{
    LOG(info) << "test_parametric" ;

    float radius = 100.f ; 
    float z1    = -0.1f ; 
    float z2    =  0.1f ; 

    ndisc ds = make_disc(radius,z1,z2); 

    // hmm need flexibility wrt par steps, only need one step for body ?

    unsigned nsurf = ds.par_nsurf();
    assert(nsurf == 3);

    unsigned nu = 5 ; 
    unsigned nv = 5 ; 

    for(unsigned s=0 ; s < nsurf ; s++)
    {
        std::cout << " surf : " << s << std::endl ; 

        for(unsigned u=0 ; u <= nu ; u++){
        for(unsigned v=0 ; v <= nv ; v++)
        {
            nuv uv = make_uv(s,u,v,nu,nv );

            glm::vec3 p = ds.par_pos(uv);

            std::cout 
                 << " s " << std::setw(3) << s  
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

    test_parametric();

    return 0 ; 
} 
