#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>
#include <map>

class Shdr ;


class Prog {
   public:
      Prog(const char* basedir, const char* tag, bool live=false);

      void createAndLink();
      void Summary(const char* msg);
      void Print(const char* msg);
      GLuint getId();

      // required attributes/uniforms cause exits when not found
      // non-required return -1 when not found just like 
      // glGetUniformLocation glGetAttribLocation when no such active U or A 
      GLint attribute(const char* name, bool required=true);
      GLint uniform(const char* name, bool required=true);

   private:
      enum Obj_t { Uniform=0, Attribute=1 } ;

      void setup();
      void examine(const char* tagdir);
      void addShader(const char* path, GLenum type);
      void traverseLocation(Obj_t obj, GLenum type,  const char* name, bool print);
      void traverseActive(Obj_t obj, bool print);
      void printStatus();

      static const char* GL_type_to_string(GLenum type); 

   private:
      void link();
      void validate();
      void collectLocations();
      void dumpUniforms();
      void dumpAttributes();
      void _print_program_info_log();

   private:
      bool                     m_live ;
      char*                    m_tagdir ;
      GLuint                   m_id ;
      std::vector<std::string> m_names;
      std::vector<GLenum>      m_codes;
      std::vector<Shdr*>       m_shaders ;

      typedef std::map<std::string, GLuint> ObjMap_t ; 
      typedef std::map<std::string, GLenum> TypMap_t ; 

      ObjMap_t  m_attributes ;
      ObjMap_t  m_uniforms   ;
      TypMap_t  m_atype ;
      TypMap_t  m_utype ;

};

inline GLuint Prog::getId()
{
    return m_id ; 
}



