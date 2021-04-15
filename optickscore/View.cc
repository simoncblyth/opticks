/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <cstdio>
#include <cmath>
#include <sstream>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "NGLM.hpp"
#include "NPY.hpp"

// npy-
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

// okc-
#include "View.hh"
#include "Ctrl.hh"

#include "PLOG.hh"

const plog::Severity View::LEVEL = PLOG::EnvLevel("View", "DEBUG"); 

const char* View::STANDARD_ = "STANDARD" ; 
const char* View::FLIGHTPATH_ = "FLIGHTPATH" ; 
const char* View::INTERPOLATED_ = "INTERPOLATED" ;   // rename to BOOKMARKS ?
const char* View::ORBITAL_ = "ORBITAL" ; 
const char* View::TRACK_ = "TRACK" ; 

const char* View::getTypeName() const
{
    return TypeName(m_type); 
}

const char* View::TypeName( View_t v )
{
    const char* s = NULL ; 
    switch(v)
    {
        case STANDARD     : s = STANDARD_     ; break ; 
        case FLIGHTPATH   : s = FLIGHTPATH_   ; break ; 
        case INTERPOLATED : s = INTERPOLATED_ ; break ; 
        case ORBITAL      : s = ORBITAL_ ; break ; 
        case TRACK        : s = TRACK_ ; break ; 
        default:            s = NULL ; break ; 
    }
    assert( s ) ; 
    return s ; 
}



const char* View::PREFIX = "view" ;
const char* View::getPrefix()
{
   return PREFIX ; 
}

const char* View::EYE = "eye" ;
const char* View::LOOK = "look" ;
const char* View::UP = "up" ;


/////////////// NConfigurable protocol START

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

    LOG(LEVEL) 
        << " name [" << name << "]" 
        << " value_ [" << value_ << "]"
        ; 

    set(name, value);
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

/////////////// NConfigurable protocol END



void View::configureF(const char* , std::vector<float>  )
{
}

void View::configureI(const char* , std::vector<int>  )
{
}

void View::configureS(const char* name, std::vector<std::string> values)
{
    if(values.empty()) return ;
    std::string last = values.back();

    LOG(LEVEL) 
        << " name [" << name << "]" 
        << " last [" << last << "]"
        ; 

    set(name, last);
}






View* View::FromArrayItem( NPY<float>* flightpath, unsigned i )
{
    assert( flightpath->hasShape( -1, 4, 4) ) ; 

    glm::vec4 eye  = flightpath->getQuad_(i, 0) ;   
    glm::vec4 look = flightpath->getQuad_(i, 1) ;   
    glm::vec4 up   = flightpath->getQuad_(i, 2) ;   
    glm::vec4 vctrl = flightpath->getQuad_(i, 3) ;   

    View* v = new View ; 
    v->setEye(eye);       
    v->setLook(look);       
    v->setUp(up); 

    Ctrl ctrl(glm::value_ptr(vctrl), 4);
    //std::string cmds = ctrl.getCommands() ; 
    v->setCmds(ctrl.cmds);      
    v->setNumCmds(ctrl.num_cmds); 

    return v ; 
}


View::View(View_t type)  : m_type(type), m_num_cmds(0) 
{
    home();

    m_axes.push_back(glm::vec4(0,1,0,0));
    m_axes.push_back(glm::vec4(0,0,1,0));
    m_axes.push_back(glm::vec4(1,0,0,0));
}

void View::setNumCmds(unsigned num_cmds)
{
    m_num_cmds = num_cmds ; 
}

void View::setCmds(const std::vector<std::string>& cmds)
{
    assert( cmds.size() == 8 ); 
    for(unsigned i=0 ; i < cmds.size() ; i++ )
    {
        m_cmds[i] = cmds[i] ; 
    }
}

const std::string& View::getCmd(unsigned i) const 
{
    assert( i < 8 ); 
    return m_cmds[i] ; 
}
bool View::hasCmds() const
{ 
    return m_num_cmds > 0  ;  
}




View::~View()
{
}


bool View::isStandard()
{
    return m_type == STANDARD ; 
}
bool View::isInterpolated()
{
    return m_type == INTERPOLATED || m_type == FLIGHTPATH ; 
}
bool View::isOrbital()
{
    return m_type == ORBITAL ; 
}
bool View::isTrack()
{
    return m_type == TRACK ; 
}
bool View::isFlightPath()
{
    return m_type == FLIGHTPATH ; 
}








bool View::hasChanged()
{
    return m_changed ; 
}

void View::setChanged(bool changed)
{
    m_changed = changed ; 
}

void View::home()
{
    m_changed = true ; 
    m_eye.x = -1.f ; 
    m_eye.y = -1.f ; 
    m_eye.z =  0.f ;

    m_look.x =  0.f ; 
    m_look.y =  0.f ; 
    m_look.z =  0.f ;

    m_up.x =  0.f ; 
    m_up.y =  0.f ; 
    m_up.z =  1.f ;
}

void View::setEye( float _x, float _y, float _z)
{
    m_eye.x = _x ;  
    m_eye.y = _y ;  
    m_eye.z = _z ;  

    handleDegenerates();
    m_changed = true ; 

#ifdef VIEW_DEBUG
    printf("View::setEye %10.3f %10.3f %10.3f \n", _x, _y, _z);
#endif
}  


void View::setEyeX(float _x)
{
    m_eye.x = _x ;   
}
void View::setEyeY(float _y)
{
    m_eye.y = _y ;   
}
void View::setEyeZ(float _z)
{
    m_eye.z = _z ;   
}


float View::getEyeX() const
{
    return m_eye.x ;   
}
float View::getEyeY() const
{
    return m_eye.y ;   
}
float View::getEyeZ() const
{
    return m_eye.z ;   
}




void View::setLook(float _x, float _y, float _z)
{
    m_look.x = _x ;  
    m_look.y = _y ;  
    m_look.z = _z ;  

    handleDegenerates();
    m_changed = true ; 

#ifdef VIEW_DEBUG
    printf("View::setLook %10.3f %10.3f %10.3f \n", _x, _y, _z);
#endif
}

void View::setUp(  float _x, float _y, float _z)
{
    m_up.x = _x ;  
    m_up.y = _y ;  
    m_up.z = _z ;  

    handleDegenerates();
    m_changed = true ; 
} 

void View::setLook(glm::vec4& look)
{
    setLook(look.x, look.y, look.z );
}
void View::setEye(glm::vec4& eye)
{
    setEye(eye.x, eye.y, eye.z );
}
void View::setUp(glm::vec4& up)
{
    setUp(up.x, up.y, up.z );
}

//void View::setCtrl(const glm::vec4& ctrl)
//{
//}





void View::Print(const char* )
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
    // no need to override in InterpolateView as invoked only other overridden methods
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


void View::handleDegenerates()
{
   // invoked by setEye, so handle here problematic viewpoints
   glm::vec3 gaze = glm::normalize(m_look - m_eye) ; 
   float eul = glm::length(glm::cross(gaze, m_up));
   if(eul==0.f)
   {
//#ifdef VIEW_DEBUG
       LOG(warning) << "View::handleDegenerates looking for ne changing up axis " ; 
//#endif
       for(unsigned int i=0 ; i < m_axes.size() ; i++)
       {
            glm::vec4 axis = m_axes[i] ; 
            float aul = glm::length(glm::cross(gaze, glm::vec3(axis)));
            if(aul > 0.f)
            {
                  setUp(axis);
//#ifdef VIEW_DEBUG
                  LOG(warning) << "View::handleDegenerates picked new up axis " << i ; 
//#endif
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
      

    /*
          glm::mat4 r ;         // <-- DONT DO THIS  
          glm::mat4 r(1.0f) ;  // DO THIS 

     The uninitialized fourth row was the cause of several days of 
     driver reinstalls and bug hunting and development of minimal reproducers
     that never did.
             OGLRap_GLFW_OpenGL_Linux_display_issue_with_new_driver.rst

    */
 
    //glm::mat4 r ;   // THIS WAS THE CAUSE OF : OGLRap_GLFW_OpenGL_Linux_display_issue_with_new_driver.rst
    glm::mat4 r(1.0f) ; //    

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


void View::reset()
{
   // do nothing default, overridden in InterpolatedView
}
void View::tick()
{
    LOG(LEVEL) << " doing nothing " ;  
   // do nothing default, overridden in InterpolatedView
}
void View::nextMode(unsigned int)
{
   // do nothing default, overridden in InterpolatedView
}
void View::commandMode(const char* )
{
   // do nothing default, overridden in InterpolatedView
}




bool View::isActive()
{
   return false ; 
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



