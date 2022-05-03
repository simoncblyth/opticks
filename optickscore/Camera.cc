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

#include <cmath> 
#include <cstdio>
#include <csignal>

#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>

#include "SSys.hh"

// npy-
#include "NGLM.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

// okc-
#include "OpticksConst.hh"
#include "Camera.hh"
#include "PLOG.hh"


const plog::Severity Camera::LEVEL = PLOG::EnvLevel("Camera", "DEBUG") ; 


// Unexplained interference between plog and Camera..
// will not compile on MSVC unless change 
//
//         NEAR -> NEAR_
//         near -> near_
//         FAR -> FAR_
//         far -> far_


const char* Camera::PREFIX = "camera" ;
const char* Camera::PRINT    = "print" ;

const char* Camera::NEAR_     = "near" ;
const char* Camera::FAR_      = "far" ;
const char* Camera::ZOOM     = "zoom" ;
const char* Camera::SCALE     = "scale" ;

const char* Camera::TYPE   = "type" ;         // formerly PARALLEL/parallel


const char* Camera::getPrefix()
{
   return PREFIX ; 
}


bool Camera::accepts(const char* name)
{
    return 
          strcmp(name, NEAR_) == 0  ||
          strcmp(name, FAR_ ) == 0  || 
          strcmp(name, SCALE ) == 0  || 
          strcmp(name, ZOOM) == 0  ;
}  



std::vector<std::string> Camera::getTags()
{
    std::vector<std::string> tags ;
    tags.push_back(NEAR_);
    tags.push_back(FAR_);
    tags.push_back(ZOOM);
    tags.push_back(SCALE);
    return tags ; 
}

std::string Camera::get(const char* name)
{
    float v(0.f) ; 

    if(     strcmp(name,NEAR_)==0)    v = getNear();
    else if(strcmp(name,FAR_)== 0 )   v = getFar();
    else if(strcmp(name,ZOOM)== 0 )   v = getZoom();
    else if(strcmp(name,SCALE)== 0 )  v = getScale();
    else
         printf("Camera::get bad name %s\n", name);

    return gformat(v);
}

void Camera::set(const char* name, std::string& s)
{
    float v = gfloat_(s); 

    if(     strcmp(name,NEAR_)==0)   setNear(v);
    else if(strcmp(name,FAR_)== 0 )  setFar(v);
    else if(strcmp(name,ZOOM)== 0 )  setZoom(v);
    else if(strcmp(name,SCALE)== 0 ) setScale(v);
    else
         printf("Camera::set bad name %s\n", name);
}


void Camera::configure(const char* name, const char* val_)
{
    std::string val(val_);
    configure(name, gfloat_(val));
}

void Camera::configure(const char* name, float value)
{
    if(       strcmp(name, ZOOM)  ==  0)  setZoom(value);
    else if( strcmp(name,  SCALE) ==  0)  setScale(value);
    else if( strcmp(name,  NEAR_) ==  0)  setNear(value);
    else if( strcmp(name,  FAR_)  ==  0)  setFar(value);
    else if( strcmp(name,  TYPE)  ==  0)  setType(unsigned(value));
    else LOG(error) << " ignoring unknown parameter " << name << " : " << value ; 
}
 

void Camera::configureS(const char* , std::vector<std::string> )
{
}

void Camera::configureI(const char* name, std::vector<int> /*values*/)
{
    if( strcmp(name, PRINT) ==  0)  Print("liveline --print");
}

void Camera::configureF(const char* name, std::vector<float> values)
{

     if(values.empty())
     {
         printf("Camera::parameter_set %s no values \n", name);
     }
     else         
     {
         float vlast = values.back() ;
         LOG(LEVEL) << name << " : " << vlast ; 

#ifdef VERBOSE
         LOG(info) << "Camera::configureF"
                   << " name " << name
                   << " vals " << values.size()
                   ;

         for(size_t i=0 ; i < values.size() ; i++ ) printf("%10.3f ", values[i]);
         printf(" : vlast %10.3f \n", vlast );
#endif
         configure(name, vlast);  
     }
}
 




Camera::Camera(int width, int height, float basis ) 
    :
    m_zoom(SSys::getenvfloat("ZOOM",1.0f)),
    m_type(0),   // gets overridden from OpticksCfg default either CAMERATYPE envvar or --cameratype option
    m_changed(true)
{
    LOG(LEVEL) << "m_type (CAMERATYPE) " << m_type ;  
    bool internal = true ; 
    setSize(width, height, internal);

#ifdef __APPLE__
    setPixelFactor(2); 
#else
    setPixelFactor(1); 
#endif

    aim(basis);

    setZoomClip(0.01f, 100.f);
} 

void Camera::aim(float basis)
{
   float a_near = basis/10.f ;
   float a_far  = basis*5.f ;
   float a_scale = basis ; 

   LOG(LEVEL) 
          << "["
          << " basis " << basis
          << " a_near " << a_near
          << " a_far " << a_far
          << " a_scale " << a_scale
          ;

   //printf("Camera::aim basis %10.4f a_near %10.4f a_far %10.4f a_scale %10.4f \n", basis, a_near, a_far, a_scale );

   setBasis(basis);
   //setNearClip( a_near/10.f,  a_near*10.f) ;
   //setNearClip( a_near/10.f,  a_near*20.f) ;
   setNearClip( a_near/10.f,  a_far ) ;

   setFarClip(  a_far/10.f,   a_far*10.f );
   setScaleClip( a_scale/10.f, a_scale*10.f );

   // oct 2018 : moved setting near and far after setting of their clip ranges
   setNear( a_near );
   setFar(  a_far );


   setScale( a_scale );  // scale should be renamed to Ortho scale, as only relevant to Orthographic projection


   LOG(LEVEL) << "]" ; 

}

void Camera::setBasis(float basis)
{
    m_basis = basis ; 
}

void Camera::commandNear( const char* cmd )
{
    assert( strlen(cmd) == 2 && cmd[0] == 'N' ) ;
    int mode = (int)cmd[1] - (int)'0' ; 

    float factor = 1.f ; 
    switch(mode)
    {
        case 0: factor=1.f ; break ;
        case 1: factor=2.f ; break ;
        case 2: factor=4.f ; break ;
    }

     
    float a_far  = m_basis*5.f ;  
    float a_near = m_basis/10.f ;
    a_near /= factor ; 

    setNearClip( a_near/10.f , a_far ) ; 
    setNear( a_near ); 
    

    LOG(info) 
        << " cmd " << cmd
        << " a_near " << a_near 
        << " m_near " << m_near 
        ;
}



bool Camera::hasChanged()
{
    return m_changed ; 
}
void Camera::setChanged(bool changed)
{
    m_changed = changed ; 
}


void Camera::nextStyle(unsigned modifiers)
{
    if(modifiers & OpticksConst::e_shift)
    {
        Summary("Camera::nextStyle Summary from D+shift");
    }
    unsigned next = (getStyle() + 1) % NUM_CAMERA_STYLE ; 
    setStyle( (Style_t)next ) ; 
}

bool Camera::isOrthographic() const 
{
    return (Style_t)m_type == ORTHOGRAPHIC_CAMERA ; 
} 

bool Camera::hasNoRasterizedRender() const 
{
    return (Style_t)m_type == EQUIRECTANGULAR_CAMERA ;    // equirect is easy wih a ray tracer, difficult with a rasterizer : currently only have for 
} 

void Camera::setStyle(Style_t style)
{
    setType( unsigned(style) );   
}
Camera::Style_t Camera::getStyle() const 
{
    return (Style_t)(m_type) ;
}

/**
Camera::setType
-----------------

This is called by Opticks initialization, overriding the above ctor default, 
from Composition::setCameraType, Opticks::postconfigureComposition/Composition::setCameraType
Hence the default in OpticksCfg is what matters. 

OpticksCfg default is set via CAMERTYPE envvar or --cameratype option to override.

**/
void Camera::setType(unsigned type)
{
    m_type = type  ;
    m_changed = true ; 
    LOG(LEVEL) << " type " << m_type ; 
    //std::raise(SIGINT);   
}

void Camera::setPerspective(){  setStyle(PERSPECTIVE_CAMERA) ; }
void Camera::setOrthographic(){ setStyle(ORTHOGRAPHIC_CAMERA) ; }
void Camera::setEquirectangular(){ setStyle(EQUIRECTANGULAR_CAMERA) ; }


unsigned Camera::getType() const { 
    LOG(LEVEL) << " type " << m_type ; 
    return m_type ; 
}

void Camera::setSize(int width, int height, bool internal )
{
    // externally invoked by OpticksHub::configureCompositionSize/Composition::setSize
    m_size[0] = width ;
    m_size[1] = height ;
    m_changed = true ; 
}
void Camera::setPixelFactor(unsigned int factor)
{
    LOG(LEVEL) << " factor " << factor ; 
    m_pixel_factor = factor ; 
    m_changed = true ; 
}

unsigned int Camera::getWidth() const {  return m_size[0]; }
unsigned int Camera::getHeight() const { return m_size[1]; }
float        Camera::getAspect() const { return float(m_size[0])/float(m_size[1]) ; }   // (> 1 for landscape) 

unsigned int Camera::getPixelWidth() const {  return m_size[0]*m_pixel_factor; }
unsigned int Camera::getPixelHeight() const { return m_size[1]*m_pixel_factor; }
unsigned int Camera::getPixelFactor() const { return m_pixel_factor ; }




void Camera::near_to( float , float , float , float dy )
{
    setNear(m_near + m_near*dy );
    //printf("Camera::near_to %10.3f \n", m_near);
}
void Camera::far_to( float , float, float , float dy )
{
    setFar(m_far + m_far*dy );
    //printf("Camera::far_to %10.3f \n", m_far);
}
void Camera::zoom_to( float , float , float , float dy )
{
    setZoom(m_zoom + 30.f*dy) ;
    //printf("Camera::zoom_to %10.3f \n", m_zoom);
}
void Camera::scale_to( float , float , float , float dy )
{
    setScale(m_scale + 30.f*dy) ;
    //printf("Camera::scale_to %10.3f \n", m_scale);
}




void Camera::setNear(float near_)
{
    if(      near_ < m_nearclip[0] )  m_near = m_nearclip[0] ;
    else if( near_ > m_nearclip[1] )  m_near = m_nearclip[1] ;
    else                             m_near = near_ ;
    m_changed = true ; 
}
void Camera::setFar(float far_)
{
    if(      far_ < m_farclip[0] )  m_far = m_farclip[0] ;
    else if( far_ > m_farclip[1] )  m_far = m_farclip[1] ;
    else                           m_far = far_ ;
    m_changed = true ; 
}
void Camera::setZoom(float zoom)
{
    if(      zoom < m_zoomclip[0] )  m_zoom = m_zoomclip[0] ;
    else if( zoom > m_zoomclip[1] )  m_zoom = m_zoomclip[1] ;
    else                             m_zoom = zoom ;
    m_changed = true ; 
}

void Camera::setScale(float scale)
{
    if(      scale < m_scaleclip[0] )  m_scale = m_scaleclip[0] ;
    else if( scale > m_scaleclip[1] )  m_scale = m_scaleclip[1] ;
    else                               m_scale = scale ;
    m_changed = true ; 
}



float Camera::getBasis() const { return m_basis ; } 
float Camera::getNear() const  {  return m_near ; }
float Camera::getFar() const {   return m_far ;  }
float Camera::getQ()  const {  return m_far/m_near  ; }
float Camera::getZoom() const {  return m_zoom ; } 
float Camera::getScale() const { return isOrthographic() ? m_scale  : m_near ; }

float Camera::getDepth() const {   return m_far - m_near ; }
float Camera::getTanYfov() const { return 1.f/m_zoom ; }  // actually tan(Yfov/2)

float Camera::getTop() const {    return getScale() / m_zoom ; }
float Camera::getBottom() const { return -getScale() / m_zoom ; }
float Camera::getLeft() const {   return -getAspect() * getScale() / m_zoom ; } 
float Camera::getRight() const {  return  getAspect() * getScale() / m_zoom ; } 

void Camera::setYfov(float yfov)
{
    // setYfov(90.) -> setZoom(1.)

    // fov = 2atan(1/zoom)
    // zoom = 1/tan(fov/2)

    float pi = boost::math::constants::pi<float>() ;
    float zoom = 1.f/tan(yfov*0.5f*pi/180.f );
    setZoom( zoom );
}
float Camera::getYfov() const 
{
    float pi = boost::math::constants::pi<float>() ;
    return 2.f*atan(1.f/m_zoom)*180.f/pi ;
}





void Camera::setNearClip(float _min, float _max)
{
    m_nearclip[0] = _min ;  
    m_nearclip[1] = _max ;  
}
void Camera::setFarClip(float _min, float _max)
{
    m_farclip[0] = _min ;  
    m_farclip[1] = _max ;  
}
void Camera::setZoomClip(float _min, float _max)
{
    m_zoomclip[0] = _min ;  
    m_zoomclip[1] = _max ;  
}
void Camera::setScaleClip(float _min, float _max)
{
    m_scaleclip[0] = _min ;  
    m_scaleclip[1] = _max ;  
}





float* Camera::getNearPtr()
{
    return &m_near ;
}
float* Camera::getFarPtr()
{
    return &m_far ;
}
float* Camera::getZoomPtr()
{
    return &m_zoom ;
}
float* Camera::getScalePtr()
{
    return &m_scale ;
}
unsigned* Camera::getTypePtr()
{
    return &m_type ;
}



float Camera::getNearMin()
{
    return m_nearclip[0];
}
float Camera::getNearMax()
{
    return std::min(m_far, m_nearclip[1]);
}


float Camera::getFarMin()
{
    return std::max(m_near, m_farclip[0]);
}
float Camera::getFarMax()
{
    return m_farclip[1];
}


float Camera::getZoomMin()
{
    return m_zoomclip[0];
}
float Camera::getZoomMax()
{
    return m_zoomclip[1];
}


float Camera::getScaleMin()
{
    return m_scaleclip[0];
}
float Camera::getScaleMax()
{
    return m_scaleclip[1];
}









void Camera::Print(const char* msg) const 
{
    printf("%s type %d  near %10.3f far %10.3f zoom %10.3f scale %10.3f \n", msg, m_type, m_near, m_far, m_zoom, getScale() );
}

std::string Camera::desc(const char* msg) const 
{
    std::stringstream ss ; 
    ss 
        <<  msg 
        << " type " << m_type 
        << std::endl  
        << " width " << std::setw(5) << m_size[0]
        << " height " << std::setw(5) << m_size[1]
        << " aspect " 
        << std::setw(10) << std::fixed << std::setprecision(3) << getAspect() 
        << std::endl 
        << " near " 
        << std::setw(10) << std::fixed << std::setprecision(3) << m_near
        << " clip " 
        << std::setw(10) << std::fixed << std::setprecision(3) << m_nearclip[0]
        << std::setw(10) << std::fixed << std::setprecision(3) << m_nearclip[1]
        << std::endl 
        << " far " 
        << std::setw(10) << std::fixed << std::setprecision(3) << m_far
        << " clip " 
        << std::setw(10) << std::fixed << std::setprecision(3) << m_farclip[0]
        << std::setw(10) << std::fixed << std::setprecision(3) << m_farclip[1]
        << std::endl 
        << " scale "
        << std::setw(10) << std::fixed << std::setprecision(3) << m_scale
        << " clip " 
        << std::setw(10) << std::fixed << std::setprecision(3) << m_scaleclip[0]
        << std::setw(10) << std::fixed << std::setprecision(3) << m_scaleclip[1]
        << std::endl 
        << " zoom "
        << std::setw(10) << std::fixed << std::setprecision(3) << m_zoom
        << " clip " 
        << std::setw(10) << std::fixed << std::setprecision(3) << m_zoomclip[0]
        << std::setw(10) << std::fixed << std::setprecision(3) << m_zoomclip[1]
        << std::endl 
        << " top "
        << std::setw(10) << std::fixed << std::setprecision(3) << getTop()
        << " bot "
        << std::setw(10) << std::fixed << std::setprecision(3) << getBottom()
        << " left "
        << std::setw(10) << std::fixed << std::setprecision(3) << getLeft()
        << " right "
        << std::setw(10) << std::fixed << std::setprecision(3) << getRight()
        << " tanYfov "
        << std::setw(10) << std::fixed << std::setprecision(3) << getTanYfov()
        << std::endl 
        ;

    std::string s = ss.str(); 
    return s ; 
}


void Camera::Summary(const char* msg) const 
{
    LOG(info) << std::endl << desc(msg)  ; 

    printf("%s  type %d \n", msg, m_type );
    printf(" width %5d height %5d  aspect %10.3f \n", m_size[0], m_size[1], getAspect() );
    printf(" near %10.3f  clip %10.3f %10.3f \n", m_near, m_nearclip[0], m_nearclip[1] );
    printf(" far  %10.3f  clip %10.3f %10.3f \n", m_far , m_farclip[0], m_farclip[1] );
    printf(" scale %10.3f  clip %10.3f %10.3f \n", m_scale, m_scaleclip[0], m_scaleclip[1] );
    printf(" zoom %10.3f  clip %10.3f %10.3f \n", m_zoom, m_zoomclip[0], m_zoomclip[1] );
    printf(" top %10.3f bot %10.3f left %10.3f right %10.3f tan(yfov/2) %10.3f \n", getTop(), getBottom(), getLeft(), getRight(), getTanYfov() );

    glm::mat4 projection = getProjection();    
    print(projection, "projection");

    glm::mat4 perspective = getPerspective();    
    print(perspective, "perspective");

    glm::mat4 ortho = getOrtho();    
    print(ortho, "ortho");

    const float* ort = glm::value_ptr(ortho);
    for(unsigned i=0 ; i < 16 ; i++)
        std::cout << std::setw(3) << i << " : " << *(ort+i) << std::endl ;  

    glm::mat4 frustum = getFrustum();    
    print(frustum, "frustum");

    const float* fru = glm::value_ptr(frustum);
    for(unsigned i=0 ; i < 16 ; i++)
        std::cout << std::setw(3) << i << " : " << *(fru+i) << std::endl ;  
}     


glm::mat4 Camera::getProjection() const 
{
    return isOrthographic() ? getOrtho() : getFrustum() ; 
}


void Camera::fillZProjection(glm::vec4& zProj) const 
{
    glm::mat4 proj = getProjection() ;
    zProj.x = proj[0][2] ; 
    zProj.y = proj[1][2] ; 
    zProj.z = proj[2][2] ; 
    zProj.w = proj[3][2] ; 
}

glm::mat4 Camera::getPerspective() const 
{
    return glm::perspective(getYfov(), getAspect(), getNear(), getFar());
}
glm::mat4 Camera::getOrtho() const 
{
    return glm::ortho( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() );
}
glm::mat4 Camera::getFrustum() const 
{
    return glm::frustum( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() );
}



void Camera::getFrustumVert(std::vector<glm::vec4>& vert, std::vector<std::string>& labels) const 
{
    /*
      Near and far are defined as positive 
      but the position of near and far planes
      in the camera frame is along -ve z at 
    
              z = -near 
              z = -far 
    */

    float near_ = getNear();
    float far_ = getFar();
    float q = getQ();

    vert.push_back( glm::vec4( getLeft(),  getBottom(), -near_ , 1.f ) );
    labels.push_back("near-bottom-left");
    vert.push_back( glm::vec4( getRight(), getBottom(), -near_ , 1.f ) );
    labels.push_back("near-bottom-right");
    vert.push_back( glm::vec4(  getRight(), getTop(),    -near_ , 1.f ) );
    labels.push_back("near-top-right");
    vert.push_back( glm::vec4( getLeft(),  getTop(),    -near_ , 1.f ) );
    labels.push_back("near-top-left");

    vert.push_back( glm::vec4( q*getLeft(),  q*getBottom(), -far_  , 1.f ) );
    labels.push_back("far-bottom-left");
    vert.push_back( glm::vec4( q*getRight(), q*getBottom(), -far_  , 1.f ) );
    labels.push_back("far-bottom-right");
    vert.push_back( glm::vec4( q*getRight(), q*getTop(),    -far_  , 1.f ) );
    labels.push_back("far-top-right");
    vert.push_back( glm::vec4( q*getLeft(),  q*getTop(),    -far_  , 1.f ) );
    labels.push_back("far-top-left");
}



