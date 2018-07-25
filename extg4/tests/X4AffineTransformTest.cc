#include <iostream>
#include "X4AffineTransform.hh"

#include "OPTICKS_LOG.hh"

#include "GLMFormat.hpp"
#include "NGLMExt.hpp"


glm::mat4* make_trs(unsigned idx)
{
    // create glm::mat4 transform

    glm::mat4* t = NULL ; 

    if( idx == 0 )
    {
         glm::vec3 tlat(10,1,3);
         glm::vec4 axis_angle(0,0,1,45.); 
         glm::vec3 scal(1,1,1);
         glm::mat4 trs = nglmext::make_transform("trs", tlat, axis_angle, scal );
         t = new glm::mat4(trs) ;
    } 
    else if( idx == 1 )
    {
         glm::vec3 tlat(10,1,3);
         glm::vec4 axis_angle(1,1,1,45.); 
         glm::vec3 scal(1,1,1);
         glm::mat4 trs = nglmext::make_transform("trs", tlat, axis_angle, scal );
         t = new glm::mat4(trs) ;
    }
    return t ; 
}

void test_FromGLM(const glm::mat4& trs )
{
    X4AffineTransform xaf = X4AffineTransform::FromGLM(trs); 
    std::cout << xaf.tla << std::endl ;  
    std::cout << xaf.rot << std::endl ;  
}

void test_getRotation(const glm::mat4& trs )
{
    X4AffineTransform xaf = X4AffineTransform::FromGLM(trs); 
    std::cout << xaf.tla << std::endl ;  
    std::cout << xaf.rot << std::endl ;  

    G4RotationMatrix rot0 = xaf.getRotation_0(); 
    G4RotationMatrix rot = xaf.getRotation(); 
    std::cout << "rot0:" << rot0 << std::endl ;  
    std::cout << " rot:" << rot << std::endl ;  

    std::cout << xaf.getRotationCode("rotgen") << std::endl ;     
}

#define G4ROTATION( xx, xy, xz, yx, yy, yz, zx, zy, zz) (G4RotationMatrix(G4ThreeVector((xx),(yx),(zx)),G4ThreeVector((xy),(yy),(zy)),G4ThreeVector((xz),(yz),(zz))))

void test_gen()
{
    G4RotationMatrix* rotgen = new G4RotationMatrix(G4ThreeVector(0.707107,-0.707107,0.000000),G4ThreeVector(0.707107,0.707107,0.000000),G4ThreeVector(0.000000,0.000000,1.000000)); 
    std::cout << " rotgen " << *rotgen << std::endl ; 

    G4RotationMatrix* rotgen2 = new G4RotationMatrix(G4ThreeVector(0.804738,-0.310617,0.505879),G4ThreeVector(0.505879,0.804738,-0.310617),G4ThreeVector(-0.310617,0.505879,0.804738)); 
    std::cout << " rotgen2 " << *rotgen2 << std::endl ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    for( unsigned idx=0 ; idx < 2 ; idx++)
    {
        glm::mat4* trs = make_trs(idx);
        std::cout << gpresent("trs", *trs) << std::endl ; 
        test_FromGLM(*trs); 
        test_getRotation(*trs);
    } 

    test_gen(); 

    return 0 ; 
}
