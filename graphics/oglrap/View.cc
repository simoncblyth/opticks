#include "View.hh"
#include "Animator.hh"

#include "stdio.h"

// npy-
#include "NLog.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"


#include <glm/glm.hpp>  
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>

#include <sstream>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>


#ifdef GUI_
#include <imgui.h>
#endif


const char* View::EYE = "eye" ;
const char* View::LOOK = "look" ;
const char* View::UP = "up" ;


std::vector<std::string> View::getTags()
{
    std::vector<std::string> tags ;
    tags.push_back(EYE);
    tags.push_back(LOOK);
    tags.push_back(UP);
    return tags ; 
}


bool View::accepts(const char* name)
{
    return 
         strcmp(name,EYE)==0  ||
         strcmp(name,LOOK)==0 ||
         strcmp(name,UP)==0 ;
}


void View::configure(const char* name, const char* value_)
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
    glm::vec3 v = gvec3(_xyz);
    if(     strcmp(name,EYE)==0)    setEye(v.x,v.y,v.z);
    else if(strcmp(name,LOOK)== 0 ) setLook(v.x,v.y,v.z);
    else if(strcmp(name,UP)== 0 )   setUp(v.x,v.y,v.z);
    else
         printf("View::set bad name %s\n", name);
}

std::string View::get(const char* name)
{
    glm::vec4 v ; 
    if(     strcmp(name,EYE)==0)    v = getEye();
    else if(strcmp(name,LOOK)== 0 ) v = getLook();
    else if(strcmp(name,UP)== 0 )   v = getUp();
    else
         printf("View::get bad name %s\n", name);

    glm::vec3 v3(v); 
    return gformat(v3);
}


void View::Print(const char* msg)
{
    print(getEye(), getLook(), getUp() , "eye/look/up");
}

void View::nextMode(unsigned int modifiers)
{
    m_animator->nextMode(modifiers);
}

void View::initAnimator()
{
    m_animator = new Animator(&m_eye_phase, 200, -1.f, 1.f ); 
    m_animator->setModeRestrict(Animator::NORM);  // only OFF and SLOW 
    //m_animator->Summary("View::initAnimator");
}

void View::tick()
{
    bool bump(false);
    if(m_animator->step(bump)) setEyePhase(m_eye_phase);
}

void View::gui()
{
#ifdef GUI_
    if(ImGui::Button("home")) home();
    ImGui::SliderFloat3("eye",  getEyePtr(),  -1.0f, 1.0f);
    ImGui::SliderFloat3("look", getLookPtr(), -1.0f, 1.0f);
    ImGui::SliderFloat3("up",   getUpPtr(), -1.0f, 1.0f);

    updateEyePhase();
    if(ImGui::SliderFloat("eyephase", &m_eye_phase,  -1.f, 1.f)) setEyePhase(m_eye_phase);

    if(m_animator->gui("animate eyephase", "%0.3f", 1.0f)) setEyePhase(m_eye_phase) ;

#endif    
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


void View::updateEyePhase()
{
   // atan2 : Principal arc tangent of y/x, in the interval [-pi,+pi] radians 
   // so eye phase range is -1:1
    m_eye_phase = atan2(m_eye.y,m_eye.x)/(1.0f*M_PI);
    
    // TODO: rotate about up, not always z

   // invoked by setEye, so handle here problematic viewpoints
   glm::vec3 gaze = glm::normalize(m_look - m_eye) ; 
   float eul = glm::length(glm::cross(gaze, m_up));

   if(eul==0.f)
   {
       LOG(warning) << "View::updateEyePhase new viewpoint causes degeneracy, so changing up axis " ; 
       for(unsigned int i=0 ; i < m_axes.size() ; i++)
       {
            glm::vec4 axis = m_axes[i] ; 
            float aul = glm::length(glm::cross(gaze, glm::vec3(axis)));
            if(aul > 0.f)
            {
                  setUp(axis);
                  LOG(warning) << "View::updateEyePhase picked new up axis " << i ; 
                  break ; 
            }
        }
   }




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
    return glm::vec4(m_eye.x, m_eye.y, m_eye.z,1.0f);
}   
glm::vec4 View::getLook()
{
    return glm::vec4(m_look.x, m_look.y, m_look.z,1.0f);
}   
glm::vec4 View::getUp()
{
    return glm::vec4(m_up.x, m_up.y, m_up.z, 0.0f); // direction, not position so w=0
}   



float* View::getEyePtr()
{
    return glm::value_ptr(m_eye);
}
float* View::getLookPtr()
{
    return glm::value_ptr(m_look);
}
float* View::getUpPtr()
{
    return glm::value_ptr(m_up);
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
    return glm::vec4( m_look.x - m_eye.x, m_look.y - m_eye.y , m_look.z - m_eye.z, 0.0f );
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



