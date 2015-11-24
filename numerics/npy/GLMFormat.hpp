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
       std::string format(unsigned int u);
       std::string format(const glm::vec3& v);
       std::string format(const glm::vec4& v);
       std::string format(const glm::ivec4& v);
       std::string format(const glm::uvec4& v);
       std::string format(const glm::ivec3& v);
       std::string format(const glm::quat& q);
       std::string format(const glm::mat4& m);

   public:
       float         float_(const std::string& s );
       int           int_(const std::string& s );
       unsigned int  uint_(const std::string& s );
       glm::vec4 vec4(const std::string& s );
       glm::ivec4 ivec4(const std::string& s );
       glm::uvec4 uvec4(const std::string& s );
       glm::ivec3 ivec3(const std::string& s );
       glm::vec3 vec3(const std::string& s );
       glm::quat quat(const std::string& s );
       glm::mat4 mat4(const std::string& s, bool flip=false);

   private:
       std::ostringstream m_ss  ;

};



std::string gformat(float f);
std::string gformat(int i);
std::string gformat(unsigned int i);

std::string gformat(const glm::vec3& v );
std::string gformat(const glm::vec4& v );
std::string gformat(const glm::ivec4& v );
std::string gformat(const glm::uvec4& v );
std::string gformat(const glm::ivec3& v );
std::string gformat(const glm::quat& q );
std::string gformat(const glm::mat4& m );

float       gfloat_(const std::string& s );
int           gint_(const std::string& s );
unsigned int guint_(const std::string& s );
glm::vec3   gvec3(const std::string& s );
glm::vec4   gvec4(const std::string& s );
glm::ivec4  givec4(const std::string& s );
glm::uvec4  guvec4(const std::string& s );
glm::ivec3  givec3(const std::string& s );
glm::quat   gquat(const std::string& s );
glm::mat4   gmat4(const std::string& s, bool flip=false);

std::string gpresent(const glm::vec4& v, unsigned int prec=3, unsigned int wid=10);




