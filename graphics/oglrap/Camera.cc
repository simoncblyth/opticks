#include "Camera.hh"

// npy-
#include "GLMPrint.hpp"

#include "stdio.h"
#include <glm/gtc/matrix_transform.hpp>  
#include <boost/lexical_cast.hpp>


const char* Camera::PRINT    = "print" ;
const char* Camera::NEAR     = "near" ;
const char* Camera::FAR      = "far" ;
const char* Camera::YFOV     = "yfov" ;
const char* Camera::PARALLEL = "parallel" ;


bool Camera::accepts(const char* name)
{
    return 
          strcmp(name, NEAR) == 0  ||
          strcmp(name, FAR ) == 0  || 
          strcmp(name, YFOV) == 0  ;
}  


void Camera::configure(const char* name, const char* value_)
{
    float value = boost::lexical_cast<float>(value_); 
    configure(name, value);
}


void Camera::configure(const char* name, float value)
{
    if(      strcmp(name, YFOV) ==  0)      setYfov(value);
    else if( strcmp(name, NEAR) ==  0)      setNear(value);
    else if( strcmp(name, FAR) ==  0)       setFar(value);
    else if( strcmp(name, PARALLEL) ==  0)  setParallel( value==0.f ? false : true );
    else
        printf("Camera::configureF ignoring unknown parameter %s : %10.3f \n", name, value); 
}
 





void Camera::configureS(const char* name, std::vector<std::string> values)
{
}

void Camera::configureI(const char* name, std::vector<int> values)
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
         printf("Camera::parameter_set %s : %lu values : ", name, values.size());
         for(size_t i=0 ; i < values.size() ; i++ ) printf("%10.3f ", values[i]);
         printf(" : vlast %10.3f \n", vlast );
#endif
         configure(name, vlast);  
     }
}
 


void Camera::Print(const char* msg)
{
    printf("%s parallel %d  near %10.3f far %10.3f yfov %10.3f \n", msg, m_parallel, m_near, m_far, m_yfov );
}


void Camera::Summary(const char* msg)
{
    printf("%s  parallel %d \n", msg, m_parallel );
    printf(" width %5d height %5d  aspect %10.3f \n", m_size[0], m_size[1], getAspect() );
    printf(" near %10.3f  clip %10.3f %10.3f \n", m_near, m_nearclip[0], m_nearclip[1] );
    printf(" far  %10.3f  clip %10.3f %10.3f \n", m_far , m_farclip[0], m_farclip[1] );
    printf(" yfov %10.3f  clip %10.3f %10.3f \n", m_yfov, m_yfovclip[0], m_yfovclip[1] );
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
glm::mat4 Camera::getPerspective()
{
    return glm::perspective(getYfov(), getAspect(), getNear(), getFar());
}
glm::mat4 Camera::getOrtho()
{
    return glm::ortho( getLeft(), getRight(), getBottom(), getTop() );
}
glm::mat4 Camera::getFrustum()
{
    return glm::frustum( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() );
}



