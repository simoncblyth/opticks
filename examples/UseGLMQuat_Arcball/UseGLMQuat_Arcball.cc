/**
UseGLMQuat_Arcball.cc
======================

::

   ~/o/examples/UseGLMQuat_Arcball/UseGLMQuat_Arcball.sh 


* https://research.cs.wisc.edu/graphics/Courses/559-f2001/Examples/Gl3D/arcball-gems.pdf
* ~/opticks_refs/ken_shoemake_arcball_rotation_control_gem.pdf 

* http://www.talisman.org/~erlkonig/misc/shoemake92-arcball.pdf
* ~/opticks_refs/shoemake92-arcball.pdf


* :google:`ken shoemake arcball rotation control gem`


Initial impl that the below Arcball started from 
-------------------------------------------------

* https://github.com/Twinklebear/arcball-cpp
* https://github.com/Twinklebear/arcball-cpp/blob/master/arcball_camera.h
* https://github.com/Twinklebear/arcball-cpp/blob/master/arcball_camera.cpp
* https://github.com/Twinklebear/arcball-cpp/blob/master/example/example.cpp


Other impls/refs to compare with 
---------------------------------

* https://github.com/iauns/cpm-arc-ball/blob/master/arc-ball/ArcBall.hpp

* https://github.com/oguz81/ArcballCamera/blob/main/arcball_paper.pdf

* https://www.khronos.org/opengl/wiki/Object_Mouse_Trackball

Comparison of rotation interface techniques
--------------------------------------------

* https://vvise.iat.sfu.ca/user/data/papers/comparerotation.pdf



**/


#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <glm/glm.hpp>
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>


struct Arcball
{
    template<typename T>
    static std::string Desc(const T* tt, int num)
    {
        std::stringstream ss ; 
        for(int i=0 ; i < num ; i++) 
            ss  
                << ( i % 4 == 0 && num > 4 ? ".\n" : "" ) 
                << " " << std::fixed << std::setw(10) << std::setprecision(4) << tt[i] 
                << ( i == num-1 && num > 4 ? ".\n" : "" ) 
                ;   

        std::string s = ss.str(); 
        return s ; 
    }

    static glm::quat screen_to_arcball(const glm::vec2& p)
    {
        const float dist = glm::dot(p, p);
        // If we're on/in the sphere return the point on it
        if (dist <= 1.f) {
            return glm::quat(0.0, p.x, p.y, std::sqrt(1.f - dist));
        } else {
            // otherwise we project the point onto the sphere
            const glm::vec2 proj = glm::normalize(p);
            return glm::quat(0.0, proj.x, proj.y, 0.f);
        }
    }

    glm::mat4 center_translation ;
    glm::mat4 translation ;
    glm::quat rotation ;
    glm::mat4 camera ;
    glm::mat4 inv_camera;

    Arcball( const glm::vec3& eye, 
             const glm::vec3& center,
             const glm::vec3& up )
    {
        const glm::vec3 dir = center - eye;
        glm::vec3 z_axis = glm::normalize(dir);
        glm::vec3 x_axis = glm::normalize(glm::cross(z_axis, glm::normalize(up)));
        glm::vec3 y_axis = glm::normalize(glm::cross(x_axis, z_axis));
        x_axis = glm::normalize(glm::cross(z_axis, y_axis));  

        center_translation = glm::inverse(glm::translate(glm::mat4(1.f), center));
        translation = glm::translate(glm::mat4(1.f), glm::vec3(0.f, 0.f, -glm::length(dir)));   
        rotation = glm::normalize(glm::quat_cast(glm::transpose(glm::mat3(x_axis, y_axis, -z_axis))));

        update_camera();
    }

    void update_camera()
    {
        camera = translation * glm::mat4_cast(rotation) * center_translation;
        inv_camera = glm::inverse(camera);
    }

    void rotate(glm::vec2 prev_mouse, glm::vec2 cur_mouse)
    {
        // Clamp mouse positions to stay in NDC
        cur_mouse = glm::clamp(cur_mouse, glm::vec2{-1, -1}, glm::vec2{1, 1});
        prev_mouse = glm::clamp(prev_mouse, glm::vec2{-1, -1}, glm::vec2{1, 1});

        const glm::quat mouse_cur_ball = screen_to_arcball(cur_mouse);
        const glm::quat mouse_prev_ball = screen_to_arcball(prev_mouse);

        rotation = mouse_cur_ball * mouse_prev_ball * rotation;
        update_camera();
    }

    void pan(glm::vec2 mouse_delta)
    {
        const float zoom_amount = std::abs(translation[3][2]);
        glm::vec4 motion(mouse_delta.x * zoom_amount, mouse_delta.y * zoom_amount, 0.f, 0.f);
        // Find the panning amount in the world space
        motion = inv_camera * motion;

        center_translation = glm::translate(glm::mat4(1.f), glm::vec3(motion)) * center_translation;
        update_camera();
    }

    void zoom(const float zoom_amount)
    {
        const glm::vec3 motion(0.f, 0.f, zoom_amount);
        translation = glm::translate(glm::mat4(1.f), motion) * translation;
        update_camera();
    }

    glm::vec3 eye() const
    {
        return glm::vec3{inv_camera * glm::vec4{0, 0, 0, 1}};
    }

    glm::vec3 dir() const
    {
        return glm::normalize(glm::vec3{inv_camera * glm::vec4{0, 0, -1, 0}});
    }

    glm::vec3 up() const
    {
        return glm::normalize(glm::vec3{inv_camera * glm::vec4{0, 1, 0, 0}});
    }

    std::string desc() const
    {
        std::stringstream ss ; 
        ss << "Arcball::desc" << std::endl 
           << " center_translation " << std::endl 
           << Desc<float>(glm::value_ptr(center_translation),16)
           << " translation " << std::endl 
           << Desc<float>(glm::value_ptr(translation),16)
           << " rotation " << std::endl 
           << glm::to_string(rotation) << std::endl 
           << " rotation(raw) " << std::endl 
           << Desc<float>(glm::value_ptr(rotation),4) << std::endl
           << " camera " << std::endl 
           << Desc<float>(glm::value_ptr(camera),16)
           << " inv_camera " << std::endl 
           << Desc<float>(glm::value_ptr(inv_camera),16)
           << std::endl
           ;  
        std::string str = ss.str(); 
        return str ; 
    } 
}; 


int main()
{
    glm::vec3 eye(   0.f,  0.f,  10.f ); 
    glm::vec3 center(0.f,  0.f,  0.f ); 
    glm::vec3 up(    0.f,  1.f,  0.f ); 

    Arcball ball(eye, center, up) ; 

    std::cout << ball.desc() ; 


    return 0 ; 
}
