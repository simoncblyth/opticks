#include <iostream>
#include "X4RotationMatrix.hh"
#include "X4Transform3D.hh"

#include "GLMFormat.hpp"
#include "NGLMExt.hpp"

#include "OPTICKS_LOG.hh"


void test_machinery()
{
    float xx = 11.f ; 
    float xy = 12.f ; 
    float xz = 13.f ; 

    float yx = 21.f ; 
    float yy = 22.f ; 
    float yz = 23.f ; 

    float zx = 31.f ; 
    float zy = 32.f ; 
    float zz = 33.f ; 

    float dx = 41.f ; 
    float dy = 42.f ; 
    float dz = 43.f ; 


    X4RotationMatrix<float> xrot( xx, xy, xz , 
                                  yx, yy, yz ,
                                  zx, zy, zz ) ; 

    G4ThreeVector xtra( dx, dy, dz );    

    G4Transform3D t( xrot, xtra );
    
    glm::mat4 m = X4Transform3D::Convert(t);

    std::string dig = X4Transform3D::Digest(t) ; 


    LOG(info) << " m " << glm::to_string(m) 
              << " dig " << dig 
              ;


#ifdef X4_TRANSFORM_43
    assert( m[0][3] == dx ) ;  
    assert( m[1][3] == dy ) ;  
    assert( m[2][3] == dz ) ;  
#else
    assert( m[3][0] == dx ) ;  
    assert( m[3][1] == dy ) ;  
    assert( m[3][2] == dz ) ;  
#endif

    assert( m[0][0] == xx ) ;  
    assert( m[0][1] == xy ) ;  
    assert( m[0][2] == xz ) ;  

    assert( m[1][0] == yx ) ;  
    assert( m[1][1] == yy ) ;  
    assert( m[1][2] == yz ) ;  

    assert( m[2][0] == zx ) ;  
    assert( m[2][1] == zy ) ;  
    assert( m[2][2] == zz ) ;  

}



void test_transform_1(float ax, float ay, float az, float angle_, float tx, float ty, float tz)
{
    G4ThreeVector axis(ax,ay,az);

    float angle = angle_ * CLHEP::pi/180.f ; 

    G4RotationMatrix rot( axis, -angle ); 

    G4ThreeVector tla(tx,ty,tz);

    G4Transform3D t( rot, tla );  // rotate then translate

    glm::mat4 m = X4Transform3D::Convert(t);

    LOG(info) << glm::to_string(m) ; 

    std::cout
            << "test_transform_1 ( HUH : HAVE TO NEGATE G4RotationMatrix ANGLE TO MATCH GLM ? )" 
            << std::endl 
            << gpresent( "m", m )
            << std::endl 
            ;

}


void test_transform_0(float ax, float ay, float az, float angle_, float tx, float ty, float tz)
{
    glm::vec4 axis_angle(ax,ay,az, angle_ * CLHEP::pi/180.f );
    glm::vec3 tlat(tx,ty,tz) ; 
    glm::vec3 scal(1,1,1) ; 
    std::string order = "trs" ; 

    glm::mat4 mat(1.f) ;
    mat = glm::translate(mat, tlat );
    mat = glm::rotate(mat, axis_angle.w, glm::vec3(axis_angle) );
    // this does the translate last, despite appearances

    LOG(info) << glm::to_string(mat) ; 

/*
    glm::mat4 mat(1.f) ;
    for(unsigned i=0 ; i < order.length() ; i++)
    {   
        switch(order[i])
        {  
            case 's': mat = glm::scale(mat, scal)         ; break ; 
            case 'r': mat = glm::rotate(mat, axis_angle.w , glm::vec3(axis_angle)) ; break ;
            case 't': mat = glm::translate(mat, tlat )    ; break ;
        }  
    }
*/

    std::cout
            << "test_transform_0 " << order
            << std::endl 
            << gpresent( "axis_angle", axis_angle )
            << gpresent( "tlat", tlat )
            << gpresent( "mat", mat )
            << std::endl 
            ;

}

void test_transform_2(float ax, float ay, float az, float angle_, float tx, float ty, float tz)
{
    glm::vec3 tlat(tx,ty,tz);
    glm::vec4 trot(ax,ay,az, angle_);
    glm::vec3 tsca(1,1,1); 
 
    glm::mat4 trs = nglmext::make_transform("trs", tlat, trot, tsca );
    // trs: translate done last (despite "t" code setup coming first)  

    std::cout
            << "test_transform_2" 
            << std::endl 
            << gpresent( "tlat", tlat )
            << gpresent( "trot", trot )
            << gpresent( "tsca", tsca )
            << gpresent( "trs", trs )
            << std::endl 
            ;
}

void test_transform()
{
    float ax = 0 ; 
    float ay = 0 ; 
    float az = 1 ;

    float angle = 45.f ; 

    float tx = 100 ; 
    float ty = 0 ; 
    float tz = 0 ;

    test_transform_0(ax, ay, az,  angle, tx, ty, tz);
    test_transform_1(ax, ay, az,  angle, tx, ty, tz); 
    test_transform_2(ax, ay, az,  angle, tx, ty, tz); 

}


int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);
 
    test_machinery();
    test_transform();
 
    return 0 ; 
}
