#pragma once

#include "glm/fwd.hpp"
#include <string>
#include <sstream>

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API GLMFormat {

   public:
       GLMFormat(const char* delim="," , unsigned int precision=4);
    
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
       std::string format(const glm::mat3& m);

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
       glm::mat3 mat3(const std::string& s, bool flip=false);

   private:
       std::ostringstream m_ss  ;
       std::string        m_delim ; 

};

#include "NPY_TAIL.hh"



NPY_API std::string gformat(float f);
NPY_API std::string gformat(int i);
NPY_API std::string gformat(unsigned int i);

NPY_API std::string gformat(const glm::vec3& v );
NPY_API std::string gformat(const glm::ivec3& v );

NPY_API std::string gformat(const glm::vec4& v );
NPY_API std::string gformat(const glm::ivec4& v );
NPY_API std::string gformat(const glm::uvec4& v );

NPY_API std::string gformat(const glm::quat& q );
NPY_API std::string gformat(const glm::mat4& m );
NPY_API std::string gformat(const glm::mat3& m );

NPY_API float       gfloat_(const std::string& s );
NPY_API int           gint_(const std::string& s );
NPY_API unsigned int guint_(const std::string& s );
NPY_API glm::vec3   gvec3(const std::string& s );
NPY_API glm::vec4   gvec4(const std::string& s );
NPY_API glm::ivec4  givec4(const std::string& s );
NPY_API glm::uvec4  guvec4(const std::string& s );
NPY_API glm::ivec3  givec3(const std::string& s );
NPY_API glm::quat   gquat(const std::string& s );
NPY_API glm::mat4   gmat4(const std::string& s, bool flip=false, const char* delim=",");
NPY_API glm::mat3   gmat3(const std::string& s, bool flip=false, const char* delim=",");

NPY_API std::string gpresent(const glm::ivec4& v, unsigned wid=7);
NPY_API std::string gpresent(const glm::uvec4& v, unsigned wid=7);
NPY_API std::string gpresent(const glm::uvec3& v, unsigned wid=7);
NPY_API std::string gpresent(const glm::vec4& v, unsigned prec=3, unsigned wid=10);
NPY_API std::string gpresent(const glm::vec3& v, unsigned prec=3, unsigned wid=10);
NPY_API std::string gpresent(const glm::vec2& v, unsigned prec=3, unsigned wid=10);

NPY_API std::string gfromstring(const glm::mat4& m, bool flip=false) ;

NPY_API std::string gpresent(const char* label, const glm::mat4& m, unsigned prec=3, unsigned wid=7, unsigned lwid=10, bool flip=false );
NPY_API std::string gpresent(const char* label, const glm::mat3& m, unsigned prec=3, unsigned wid=7, unsigned lwid=10, bool flip=false );

NPY_API std::string gpresent(const char* label, const glm::ivec4& m, unsigned prec=3, unsigned wid=7, unsigned lwid=10 );
NPY_API std::string gpresent(const char* label, const glm::vec4& m, unsigned prec=3, unsigned wid=7, unsigned lwid=10 );
NPY_API std::string gpresent_(const char* label, const glm::vec4& m, unsigned prec=3, unsigned wid=7, unsigned lwid=10 );

NPY_API std::string gpresent(const char* label, const glm::vec3& m, unsigned prec=3, unsigned wid=7, unsigned lwid=10 );
NPY_API std::string gpresent(const char* label, const glm::ivec3& m, unsigned prec=3, unsigned wid=7, unsigned lwid=10 );



NPY_API std::string gpresent_label(const char* label, unsigned lwid=10 ) ; 




