#include <iostream>
#include <iomanip>

#include "GLMFormat.hpp"
#include "Nuv.hpp"
#include "NCone.hpp"

#include "PLOG.hh"


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

            glm::vec3 p = cone.par_pos_model(uv);

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




void test_getSurfacePointsAll()
{
    float r1 = 4.f ; 
    float z1 = 0.f ;
    float r2 = 2.f ; 
    float z2 = 2.f ;

    ncone cone = make_cone(r1,z1,r2,z2) ; 

    cone.verbosity = 3 ;  
    cone.pdump("make_cone(4,0,2,2)");

    unsigned level = 5 ;  // +---+---+
    int margin = 1 ;      // o---*---o
    std::vector<glm::vec3> surf ; 
    cone.getSurfacePointsAll( surf, level, margin, FRAME_LOCAL, cone.verbosity); 

    LOG(info) << "test_getSurfacePointsAll"
              << " surf " << surf.size()
              ;

    for(unsigned i=0 ; i < surf.size() ; i++ )
    {
        glm::vec3 p = surf[i]; 
        float sd = cone(p.x, p.y, p.z);

        std::cout << " p " << gpresent(p) 
                  << " sd " << sd
                  << " sd(sci) " << std::scientific << sd << std::defaultfloat 
                  << std::endl
                  ; 
    }

}






int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    //test_sdf();
    //test_parametric();
    test_getSurfacePointsAll();

    return 0 ; 
} 
