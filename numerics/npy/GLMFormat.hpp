#pragma once

#include "glm/fwd.hpp"
#include <string>
#include <sstream>


class GLMFormat {

   public:
       GLMFormat(unsigned int precision=4);
    
   public:
       std::string format(float f);
       std::string format(int i);
       std::string format(const glm::vec3& v);
       std::string format(const glm::vec4& v);
       std::string format(const glm::quat& q);

   public:
       float   float_(std::string& s );
       int       int_(std::string& s );
       glm::vec4 vec4(std::string& s );
       glm::vec3 vec3(std::string& s );
       glm::quat quat(std::string& s );
   
   private:
       std::ostringstream m_ss  ;

};



std::string gformat(float f);
std::string gformat(int i);
std::string gformat(const glm::vec3& v );
std::string gformat(const glm::vec4& v );
std::string gformat(const glm::quat& q );

float     gfloat_(std::string& s );
int         gint_(std::string& s );
glm::vec3   gvec3(std::string& s );
glm::vec4   gvec4(std::string& s );
glm::quat   gquat(std::string& s );





