// glm-;clang++ -I$(glm-dir) $(glm-sdir)/lookat.cc -o /tmp/lookat && /tmp/lookat 

#include <glm/glm.hpp>  
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <iostream>
#include <string>
#include "float.h"


void print(glm::vec4& v, const char* msg)
{
    std::cout << msg << " " << glm::to_string(v) << std::endl ;     
}

void print(glm::vec3& v, const char* msg)
{
    std::cout << msg << " " << glm::to_string(v) << std::endl ;     
}

void print(glm::mat4& m4, const char* msg)
{
    std::cout << msg << std::endl ; 


    //std::cout << glm::to_string(m4) << std::endl ; 

    std::cout << "[0] " << glm::to_string(m4[0]) << std::endl ;     
    std::cout << "[1] " << glm::to_string(m4[1]) << std::endl ;     
    std::cout << "[2] " << glm::to_string(m4[2]) << std::endl ;     
    std::cout << "[3] " << glm::to_string(m4[3]) << std::endl ;     

 
    printf("16 raw floats");
    float* f = glm::value_ptr(m4);
    for(unsigned int i=0 ; i < 16 ; i++)
    {
        if(i % 4 == 0) printf("\n"); 
        printf(" %15.3f ", *(f+i)) ;
    }
    printf("\n");



  /*
    glm::mat3 m3(m4);
    std::cout << msg << " extract 3x3 " << std::endl ; 
    std::cout << "[0] " << glm::to_string(m3[0]) << std::endl ;     
    std::cout << "[1] " << glm::to_string(m3[1]) << std::endl ;     
    std::cout << "[2] " << glm::to_string(m3[2]) << std::endl ;     

    glm::vec3 t3(m4[3]);
    std::cout << msg << " extract t3 " << std::endl ; 
    std::cout << "[3][:3] " << glm::to_string(t3) << std::endl ;     
  */ 

}

glm::mat4 view_transform( glm::vec3& eye, glm::vec3& look, glm::vec3& up, bool inverse=false)
{

    /*
    // see env/geant4/geometry/collada/g4daeview/daeutil.py
    """
    NB actual view transform in use adopts gluLookAt, this
    is here as a check and in order to obtain the inverse
    of gluLookAt

    OpenGL eye space convention with forward as -Z
    means that have to negate the forward basis vector in order 
    to create a right-handed coordinate system.

    Construct matrix using the normalized basis vectors::    

                             -Z
                       +Y    .  
                        |   .
                  EY    |  .  -EZ forward 
                  top   | .  
                        |. 
                        E-------- +X
                       /  EX right
                      /
                     /
                   +Z

    """
    */

    glm::vec3 gaze(look - eye);

    glm::vec3 forward = glm::normalize(gaze);                       // -Z
    glm::vec3 right   = glm::normalize(glm::cross(forward,up));     // +X
    glm::vec3 top     = glm::normalize(glm::cross(right,forward));  // +Y
    
    glm::mat4 r ;
    r[0] = glm::vec4( right, 0.f ); 
    r[1] = glm::vec4( top  , 0.f ); 
    r[2] = glm::vec4( -forward, 0.f ); 
  
    if(inverse)
    {
        // camera2world 
        //     un-rotate first (eye already at origin)
        //     then translate back to world  
        //
        glm::mat4 ti(glm::translate(glm::vec3(eye)));  
        return ti * r ;
    }   
    else
    {
        //
        // world2camera 
        //     must translate first putting the eye at the origin
        //     then rotate to point -Z forward
        //
        //  this is equivalent to lookAt as used by OpenGL ModelView
        //
        glm::mat4 t(glm::translate(glm::vec3(-eye)));  
        return  glm::transpose(r) * t ;  
    }
}



/*

   env/geant4/geometry/collada/g4daeview


    def _get_eye2world(self):
        return reduce(np.dot, [self.view.camera2world,
                               self.view.translate_look2eye,
                               self.trackball.rotation.T,
                               self.view.translate_eye2look,
                               self.trackball.untranslate,
                               self.upscale])
    eye2world = property(_get_eye2world)




*/

void check_lookat()
{
    glm::vec3 eye(100.f,100.f,100.f);
    glm::vec3 look(120.f,120.f,120.f);
    glm::vec3 up(0.f,1.f,0.f);

    print(eye, "eye");
    print(look, "look");
    print(up, "up");

    glm::mat4 lookat = glm::lookAt(eye, look, up);
    glm::mat4 lookat_I = glm::inverse(lookat);
    glm::mat4 lookat_T = glm::transpose(lookat);

    print(lookat,"lookat");
    //print(lookat_T,"lookat.T");
    //print(lookat_I,"lookat.I");

    glm::mat4 vt  = view_transform( eye, look, up, false);
    glm::mat4 vti = view_transform( eye, look, up, true);
    print(vt, "view_transform");
  
    glm::mat4 df = (lookat - vt)*1000.f ; 
    print(df, "(lookat-vt)*1000.");

    print(vti,"view_transform inverse");
 
    glm::mat4 vtvti = vt * vti ;
    print(vtvti,"vt * vti");

}


int test_mult()
{
    glm::mat3 r ;
    r[0] = glm::vec3(1.0,1.1,1.2); 
    r[1] = glm::vec3(2.0,2.1,2.2); 
    r[2] = glm::vec3(3.0,3.1,3.2); 

    glm::vec3 t(10.,20.,30);  // hmm not corresponding to translation
    glm::mat4 m(r) ; 
    m[3] = glm::vec4( t, 1.0 );
    
    glm::vec4 x(1.0, 0., 0., 0.);  // axes are directions not positions
    glm::vec4 y(0.0, 1., 0., 0.);
    glm::vec4 z(0.0, 0., 1., 0.);
    glm::vec4 o(0.0, 0., 0., 1.);   // origin is a position

    glm::vec4 mx = m * x ; 
    glm::vec4 my = m * y ; 
    glm::vec4 mz = m * z ; 
    glm::vec4 mo = m * o ; 

    print(m,"m");
    print(mx,"mx");
    print(my,"my");
    print(mz,"mz");
    print(mo,"mo");

    //glm::mat4 m2 = glm::translate( glm::mat4(r), t ); 
    //print(m2,"m2");  // hmm not matching 


    return 0 ;
}


void minmax(glm::mat4& m, float& mn, float& mx)
{
    float* f = glm::value_ptr(m);
    for(unsigned int i=0 ; i < 16 ; i++)
    {
         float v = *(f+i);
         if(v>mx) mx = v ;
         if(v<mn) mn = v ;
         printf(" %2d %f \n", i, v );
    }
}

float absmax(glm::mat4& m)
{
    float mn(FLT_MAX);
    float mx(-FLT_MIN);
    minmax(m, mn, mx);
    return std::max( fabs(mn), fabs(mx));
}


void test_absmax()
{
    glm::mat4 m(1.0f);

    m[0] = glm::vec4(1.0f, 1.1f, 1.2f, 1.3f);
    m[1] = glm::vec4(2.0f, 2.1f, 2.2f, 2.3f);
    m[2] = glm::vec4(3.0f, 3.1f, 3.2f, 3.3f);
    m[3] = glm::vec4(4.0f, 4.1f, 4.2f, 4.3f);

    float* f = glm::value_ptr(m);
    for(unsigned int i=0 ; i < 16 ; i++) printf(" %2d %f \n", i, *(f+i) );

    float mn(FLT_MAX);
    float mx(-FLT_MIN);
    minmax(m, mn, mx);

    float amx = absmax(m);

    printf("mn %f mx %f amx %f \n", mn, mx, amx);
}



int main()
{
    glm::vec3 t(10,20,30);
    glm::vec3 s(3);

    //glm::mat4 ts = glm::transpose(glm::translate(glm::scale(glm::mat4(1.0), s), t));
    //glm::mat4 st = glm::transpose(glm::scale(glm::translate(glm::mat4(1.0), t), s));

    glm::mat4 ts = glm::translate(glm::scale(glm::mat4(1.0), s), t);
    glm::mat4 st = glm::scale(glm::translate(glm::mat4(1.0), t), s);


    // glm sticking translation along the bottom in m[3]

    print(ts, "ts");   
    print(st, "st");   

    glm::vec4 ori(0,0,0,1);

    glm::vec4 ts_ori = ts * ori ; 
    glm::vec4 st_ori = st * ori ; 
    glm::vec4 ori_ts = ori * ts ; 
    glm::vec4 ori_st = ori * st ; 

    print(ts_ori, "ts * ori");   
    print(st_ori, "st * ori");   

    print(ori_ts, "ori * ts");   
    print(ori_st, "ori * st ");   


    return 0 ;
}

