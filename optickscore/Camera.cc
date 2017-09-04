#include <cmath> 
#include <cstdio>

#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>


// npy-
#include "NGLM.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

// okc-
#include "OpticksConst.hh"
#include "Camera.hh"
#include "PLOG.hh"

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

const char* Camera::PARALLEL = "parallel" ;



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

    if(     strcmp(name,NEAR_)==0)     v = getNear();
    else if(strcmp(name,FAR_)== 0 )    v = getFar();
    else if(strcmp(name,ZOOM)== 0 )   v = getZoom();
    else if(strcmp(name,SCALE)== 0 )  v = getScale();
    else
         printf("Camera::get bad name %s\n", name);

    return gformat(v);
}

void Camera::set(const char* name, std::string& s)
{
    float v = gfloat_(s); 

    if(     strcmp(name,NEAR_)==0)    setNear(v);
    else if(strcmp(name,FAR_)== 0 )   setFar(v);
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
    if( strcmp(name, ZOOM) ==  0)      setZoom(value);
    else if( strcmp(name, SCALE) ==  0)     setScale(value);
    else if( strcmp(name, NEAR_) ==  0)      setNear(value);
    else if( strcmp(name, FAR_) ==  0)       setFar(value);
    else if( strcmp(name, PARALLEL) ==  0)  setParallel( value==0.f ? false : true );
    else
        printf("Camera::configure ignoring unknown parameter %s : %10.3f \n", name, value); 
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
         m_zoom(1.0f),
         m_parallel(false),
         m_changed(true)
{
    setSize(width, height);
    setPixelFactor(1); 

    aim(basis);

    setZoomClip(0.01f, 100.f);
} 

void Camera::aim(float basis)
{
   float a_near = basis/10.f ;
   float a_far  = basis*5.f ;
   float a_scale = basis ; 

   //printf("Camera::aim basis %10.4f a_near %10.4f a_far %10.4f a_scale %10.4f \n", basis, a_near, a_far, a_scale );

   setBasis(basis);
   setNear( a_near );
   setFar(  a_far );

   //setNearClip( a_near/10.f,  a_near*10.f) ;
   //setNearClip( a_near/10.f,  a_near*20.f) ;
   setNearClip( a_near/10.f,  a_far ) ;

   setFarClip(  a_far/10.f,   a_far*10.f );
   setScaleClip( a_scale/10.f, a_scale*10.f );

   setScale( a_scale );  // scale should be renamed to Ortho scale, as only relevant to Orthographic projection
}

void Camera::setBasis(float basis)
{
    m_basis = basis ; 
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
    int next = (getStyle() + 1) % NUM_CAMERA_STYLE ; 
    setStyle( (Style_t)next ) ; 
}
void Camera::setStyle(Style_t style)
{
    m_parallel = int(style) == 1  ;
}
Camera::Style_t Camera::getStyle()
{
    return (Style_t)(m_parallel ? 1 : 0 ) ;
}






void Camera::setParallel(bool parallel)
{
    m_parallel = parallel ;
    m_changed = true ; 
}
bool Camera::getParallel(){ return m_parallel ; }

void Camera::setSize(int width, int height )
{
    m_size[0] = width ;
    m_size[1] = height ;
    m_aspect  = (float)width/(float)height ;   // (> 1 for landscape) 
    m_changed = true ; 
}
void Camera::setPixelFactor(unsigned int factor)
{
    m_pixel_factor = factor ; 
    m_changed = true ; 
}

unsigned int Camera::getWidth(){  return m_size[0]; }
unsigned int Camera::getHeight(){ return m_size[1]; }
float        Camera::getAspect(){ return m_aspect ; }

unsigned int Camera::getPixelWidth(){  return m_size[0]*m_pixel_factor; }
unsigned int Camera::getPixelHeight(){ return m_size[1]*m_pixel_factor; }
unsigned int Camera::getPixelFactor(){ return m_pixel_factor ; }




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
float Camera::getScale() const { return m_parallel ? m_scale  : m_near ; }

float Camera::getDepth() const {   return m_far - m_near ; }
float Camera::getTanYfov() const { return 1.f/m_zoom ; }  // actually tan(Yfov/2)

float Camera::getTop() const {    return getScale() / m_zoom ; }
float Camera::getBottom() const { return -getScale() / m_zoom ; }
float Camera::getLeft() const {   return -m_aspect * getScale() / m_zoom ; } 
float Camera::getRight() const {  return  m_aspect * getScale() / m_zoom ; } 

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
bool* Camera::getParallelPtr()
{
    return &m_parallel ;
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









void Camera::Print(const char* msg)
{
    printf("%s parallel %d  near %10.3f far %10.3f zoom %10.3f scale %10.3f \n", msg, m_parallel, m_near, m_far, m_zoom, getScale() );
}


void Camera::Summary(const char* msg)
{
    printf("%s  parallel %d \n", msg, m_parallel );
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

    glm::mat4 frustum = getFrustum();    
    print(frustum, "frustum");
}     


glm::mat4 Camera::getProjection()
{
    return m_parallel ? getOrtho() : getFrustum() ; 
}


void Camera::fillZProjection(glm::vec4& zProj)
{
    glm::mat4 proj = getProjection() ;
    zProj.x = proj[0][2] ; 
    zProj.y = proj[1][2] ; 
    zProj.z = proj[2][2] ; 
    zProj.w = proj[3][2] ; 
}

glm::mat4 Camera::getPerspective()
{
    return glm::perspective(getYfov(), getAspect(), getNear(), getFar());
}
glm::mat4 Camera::getOrtho()
{
    return glm::ortho( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() );
}
glm::mat4 Camera::getFrustum()
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




