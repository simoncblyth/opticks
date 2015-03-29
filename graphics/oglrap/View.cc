#include "View.hh"
#include "Common.hh"
#include "stdio.h"

#include <glm/glm.hpp>  
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>


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



