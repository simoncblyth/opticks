#include "Camera.hh"

// npy-
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

#include "stdio.h"
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtx/transform.hpp>
#include <boost/lexical_cast.hpp>

#ifdef GUI_
#include <imgui.h>
#endif


const char* Camera::PRINT    = "print" ;
const char* Camera::NEAR     = "near" ;
const char* Camera::FAR      = "far" ;
const char* Camera::YFOV     = "yfov" ;
const char* Camera::ZOOM     = "zoom" ;
const char* Camera::PARALLEL = "parallel" ;


bool Camera::accepts(const char* name)
{
    return 
          strcmp(name, NEAR) == 0  ||
          strcmp(name, FAR ) == 0  || 
          strcmp(name, ZOOM ) == 0  || 
          strcmp(name, YFOV) == 0  ;
}  



std::vector<std::string> Camera::getTags()
{
    std::vector<std::string> tags ;
    tags.push_back(NEAR);
    tags.push_back(FAR);
    tags.push_back(YFOV);
    tags.push_back(ZOOM);
    return tags ; 
}

std::string Camera::get(const char* name)
{
    float v ; 

    if(     strcmp(name,NEAR)==0)     v = getNear();
    else if(strcmp(name,FAR)== 0 )    v = getFar();
    else if(strcmp(name,YFOV)== 0 )   v = getYfov();
    else if(strcmp(name,ZOOM)== 0 )   v = getZoom();
    else
         printf("Camera::get bad name %s\n", name);

    return gformat(v);
}

void Camera::set(const char* name, std::string& s)
{
    float v = gfloat_(s); 

    if(     strcmp(name,NEAR)==0)    setNear(v);
    else if(strcmp(name,FAR)== 0 )   setFar(v);
    else if(strcmp(name,YFOV)== 0 )  setYfov(v);
    else if(strcmp(name,ZOOM)== 0 )  setZoom(v);
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
    if(      strcmp(name, YFOV) ==  0)      setYfov(value);
    else if( strcmp(name, ZOOM) ==  0)      setZoom(value);
    else if( strcmp(name, NEAR) ==  0)      setNear(value);
    else if( strcmp(name, FAR) ==  0)       setFar(value);
    else if( strcmp(name, PARALLEL) ==  0)  setParallel( value==0.f ? false : true );
    else
        printf("Camera::configure ignoring unknown parameter %s : %10.3f \n", name, value); 
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
 


void Camera::gui()
{
#ifdef GUI_
    float power = 2.0f ; 
    ImGui::SliderFloat("near",  &m_near, m_nearclip[0], m_nearclip[1], "%.3f", power );  
    ImGui::SliderFloat("far",   &m_far,  m_farclip[0],  m_farclip[1], "%.3f", power );
    ImGui::SliderFloat("yfov",  &m_yfov, m_yfovclip[0], m_yfovclip[1]);
    ImGui::SliderFloat("zoom",  &m_zoom, m_zoomclip[0], m_zoomclip[1]);
    ImGui::Checkbox("parallel", &m_parallel);
    if (ImGui::Button("Camera Summary")) Summary();
#endif  
 }




void Camera::Print(const char* msg)
{
    printf("%s parallel %d  near %10.3f far %10.3f yfov %10.3f zoom %10.3f \n", msg, m_parallel, m_near, m_far, m_yfov, m_zoom );
}


void Camera::Summary(const char* msg)
{
    printf("%s  parallel %d \n", msg, m_parallel );
    printf(" width %5d height %5d  aspect %10.3f \n", m_size[0], m_size[1], getAspect() );
    printf(" near %10.3f  clip %10.3f %10.3f \n", m_near, m_nearclip[0], m_nearclip[1] );
    printf(" far  %10.3f  clip %10.3f %10.3f \n", m_far , m_farclip[0], m_farclip[1] );
    printf(" yfov %10.3f  clip %10.3f %10.3f \n", m_yfov, m_yfovclip[0], m_yfovclip[1] );
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
    // TODO: find some way of scaling to make switching less jarring 
    //       Ortho tends to be in extreme closeup 
    //       
    //return m_parallel ? getOrtho() : getFrustum() ; 
    return m_parallel ? getOrthoScaled() : getFrustum() ; 
}
glm::mat4 Camera::getPerspective()
{
    return glm::perspective(getYfov(), getAspect(), getNear(), getFar());
}
glm::mat4 Camera::getOrtho()
{
    //return glm::ortho( getLeft(), getRight(), getBottom(), getTop() );
    // the form with near/far parameters is easier to adjust to get something visible
    //return glm::ortho( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() );
    return glm::ortho( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() );
}

glm::mat4 Camera::getOrthoScaled()
{
    //
    // real camera state : near, yfov, aspect, (far) 
    //
    //         top, bottom <- near * tanYfov
    //         left, right <- aspect * near * tanYfov
    //
    //  https://en.wikipedia.org/wiki/Orthographic_projection

    float sc = ((m_near + m_far)/2.f)/m_near ;     // this is closer, but need to fiddle near and far to keep visibility

    return glm::ortho( sc*getLeft(), sc*getRight(), sc*getBottom(), sc*getTop(), sc*getNear(), sc*getFar() );
}

glm::mat4 Camera::getOrthoScaled2()
{
    float sc = ((m_near + m_far)/2.f)/m_near ;     

    glm::mat4 orth = glm::ortho( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() );

    glm::vec3 factors(sc,sc,sc);

    // weak perspective projection ?

    return glm::scale( orth, factors );
}



glm::mat4 Camera::getFrustum()
{
    return glm::frustum( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() );
}











