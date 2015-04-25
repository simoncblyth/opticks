#include "View.hh"

#include "stdio.h"

// npy-
#include "GLMPrint.hpp"


#include <glm/glm.hpp>  
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>


const char* View::EYE = "eye" ;
const char* View::LOOK = "look" ;
const char* View::UP = "up" ;

bool View::accepts(const char* name)
{
    return 
         strcmp(name,EYE)==0  ||
         strcmp(name,LOOK)==0 ||
         strcmp(name,UP)==0 ;
}


void View::configureS(const char* name, const char* value_)
{
    std::string value(value_);
    set(name, value);
}

void View::configureS(const char* name, std::vector<std::string> values)
{
    if(values.empty()) return ;

    std::string last = values.back();
    set(name, last);
}




void View::set(const char* name, std::string& _xyz)
{
    std::vector<std::string> xyz;
    boost::split(xyz, _xyz, boost::is_any_of(","));

    if(xyz.size() == 3 )
    {
        float x = boost::lexical_cast<float>(xyz[0]); 
        float y = boost::lexical_cast<float>(xyz[1]); 
        float z = boost::lexical_cast<float>(xyz[2]); 

        if(     strcmp(name,EYE)==0)    setEye(x,y,z);
        else if(strcmp(name,LOOK)== 0 ) setLook(x,y,z);
        else if(strcmp(name,UP)== 0 )   setUp(x,y,z);
        else
              printf("View::configureS bad name %s\n", name);

    }
    else
    {
        printf("View::set malformed %s : %s \n", name, _xyz.c_str() );
    }
}

void View::Print(const char* msg)
{
    print(getEye(), getLook(), getUp() , "eye/look/up");
}



void View::Summary(const char* msg)
{
    printf("%s\n", msg);
    print(getEye() , "eye");
    print(getLook(),"look");
    print(getUp()  ,  "up");
}

glm::mat4 View::getLookAt(const glm::mat4& m2w, bool debug)
{
    glm::vec4 eye  = getEye(m2w); 
    glm::vec4 look = getLook(m2w); 
    glm::vec4 up   = getUp(m2w); 
    glm::mat4 lka  = glm::lookAt( glm::vec3(eye), glm::vec3(look), glm::vec3(up));

    if(debug)
    {
        printf("View::getLookAt debug\n");
        print(eye, "eye_w");
        print(look,"look_w");
        print(up,  "up_w");
        print(lka, "lka");
    }
    return lka ; 
}


void View::getFocalBasis(const glm::mat4& m2w,  glm::vec3& e, glm::vec3& u, glm::vec3& v, glm::vec3& w)
{
    glm::vec3 eye  = glm::vec3(getEye(m2w));
    glm::vec3 up   = glm::vec3(getUp(m2w));
    glm::vec3 gaze = glm::vec3(getGaze(m2w));  // look - eye

    e = eye ;    
    u = glm::normalize(glm::cross(gaze, up)); // "x" to the right
    v = glm::normalize(glm::cross(u,gaze));   // "y" to the top
    w = gaze ;                                // "-z" into target  (+z points out of screen as RHS)
}  




void View::getTransforms(const glm::mat4& m2w, glm::mat4& world2camera, glm::mat4& camera2world, glm::vec4& gaze )
{
    /*  
    See 
           env/geant4/geometry/collada/g4daeview/daeutil.py
           env/graphics/glm/lookat.cc


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

    */

    glm::vec3 eye  = glm::vec3(getEye(m2w));
    glm::vec3 up   = glm::vec3(getUp(m2w));
    glm::vec3 gze  = glm::vec3(getGaze(m2w));  // look - eye

    glm::vec3 forward = glm::normalize(gze);                        // -Z
    glm::vec3 right   = glm::normalize(glm::cross(forward,up));     // +X
    glm::vec3 top     = glm::normalize(glm::cross(right,forward));  // +Y
       
    glm::mat4 r ; 
    r[0] = glm::vec4( right, 0.f );  
    r[1] = glm::vec4( top  , 0.f );  
    r[2] = glm::vec4( -forward, 0.f );  

    glm::mat4 ti(glm::translate(glm::vec3(eye)));  

    glm::mat4 t(glm::translate(glm::vec3(-eye)));  // eye to origin


    world2camera = glm::transpose(r) * t  ;
    //
    //  must translate first putting the eye at the origin
    //  then rotate to point -Z forward
    //  this is equivalent to lookAt as used by OpenGL ModelView

    camera2world = ti * r ;
    //
    // un-rotate first (eye already at origin)
    // then translate back to world  
    // 

    gaze = glm::vec4( gze, 0.f );

    //  not normalized, vector from eye -> look 


#ifdef DEBUG
    glm::mat4 lookat = getLookAt(m2w);
    glm::mat4 diff = lookat - world2camera ; 
    print(diff, "lookat - world2camera ");
    float amx = absmax(diff)*1e6;
    printf("absmax*1e6 %f \n", amx);
#endif


}
 










/*

  glm::lookAt transforms from worldspace into OpenGL eye space 


                Y
                |   O
                |  .
                | .
                |.
                E----> X
               /
              /
             /
            Z



 /usr/local/env/graphics/glm/glm-0.9.6.3/glm/gtc/matrix_transform.inl

394         tvec3<T, P> const f(normalize(center - eye));
395         tvec3<T, P> const s(normalize(cross(f, up)));
396         tvec3<T, P> const u(cross(s, f));
397 
398         tmat4x4<T, P> Result(1);
399         Result[0][0] = s.x;
400         Result[1][0] = s.y;
401         Result[2][0] = s.z;
402         Result[0][1] = u.x;
403         Result[1][1] = u.y;
404         Result[2][1] = u.z;
405         Result[0][2] =-f.x;
406         Result[1][2] =-f.y;
407         Result[2][2] =-f.z;
408         Result[3][0] =-dot(s, eye);
409         Result[3][1] =-dot(u, eye);
410         Result[3][2] = dot(f, eye);
411         return Result;




*/










glm::vec4 View::getEye()
{
    return glm::vec4(m_eye_x, m_eye_y, m_eye_z,1.0f);
}   
glm::vec4 View::getLook()
{
    return glm::vec4(m_look_x, m_look_y, m_look_z,1.0f);
}   
glm::vec4 View::getUp()
{
    return glm::vec4(m_up_x, m_up_y, m_up_z, 0.0f); // direction, not position so w=0
}   

glm::vec4 View::getEye(const glm::mat4& m2w)
{
    return m2w * getEye();
}   
glm::vec4 View::getLook(const glm::mat4& m2w)
{
    return m2w * getLook();
}   
glm::vec4 View::getUp(const glm::mat4& m2w)
{
    return m2w * getUp(); // direction, not position so w=0
}   

glm::vec4 View::getGaze()
{
    return glm::vec4( m_look_x - m_eye_x, m_look_y - m_eye_y , m_look_z - m_eye_z, 0.0f );
}

glm::vec4 View::getGaze(const glm::mat4& m2w, bool debug)
{
    glm::vec4 a = m2w * getGaze() ;
    if(debug)
    {
        glm::vec4 b = m2w * getLook() - m2w * getEye() ;
        print(a, "View::getGaze a");
        print(b, "View::getGaze b");
    }
    return a ; 
}



