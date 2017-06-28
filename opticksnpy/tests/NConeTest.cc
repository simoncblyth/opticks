#include <iostream>
#include <iomanip>

#include "PLOG.hh"
#include "Nuv.hpp"
#include "NCone.hpp"


void test_sdf()
{
    LOG(info) << "test_sdf" ; 

    float r1 = 4.f ; 
    float z1 = 0.f ;

    float r2 = 2.f ; 
    float z2 = 2.f ;

    ncone cone = make_cone(r1,z1,r2,z2) ; 
    nnode* node = (nnode*)&cone ;

    for(float v=10. ; v >= -10. ; v-=1.f )
        std::cout 
               << " v       " << std::setw(10) << v 
               << " x       " << std::setw(10) << (*node)(v, 0, 0) 
               << " y       " << std::setw(10) << (*node)(0, v, 0) 
               << " z       " << std::setw(10) << (*node)(0, 0, v) 
               << " x(z=1)  " << std::setw(10) << (*node)(v, 0, 1.f) 
               << " y(z=1)  " << std::setw(10) << (*node)(0, v, 1.f) 
               << " x(z=-1) " << std::setw(10) << (*node)(v, 0, -1.f) 
               << " y(z=-1) " << std::setw(10) << (*node)(0, v, -1.f) 
               << std::endl ;  


}


void test_parametric()
{
    LOG(info) << "test_parametric" ; 

    float r1 = 4.f ; 
    float z1 = 0.f ;

    float r2 = 2.f ; 
    float z2 = 2.f ;

    ncone cone = make_cone(r1,z1,r2,z2) ; 
 
    unsigned nsurf = cone.par_nsurf();
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

            glm::vec3 p = cone.par_pos(uv);

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

    //test_sdf();
    test_parametric();

    return 0 ; 
} 
