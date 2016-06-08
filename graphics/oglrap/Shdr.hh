#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>

class Shdr {

   //friend class Prog ; 

   public:
       static const char* incl_prefix ; 

       // ctor only reads the file, no context needed
       Shdr(const char* path, GLenum type, const char* incl_path=NULL);

    public:
       void createAndCompile();
       void Print(const char* msg);
       GLuint getId();
 
   private:
       void setInclPath(const char* path, const char* delim=";"); // semicolon delimited list of directories to look for glsl inclusions
       std::string resolve(const char* name);
       void readFile(const char* path);
       void _print_shader_info_log();

   private:
       char*      m_path ; 
       GLenum     m_type ; 
       GLuint     m_id ;

       std::string              m_content;
       std::vector<std::string> m_lines ; 

       std::string              m_incl_path ; 
       std::vector<std::string> m_incl_dirs ; 

};


inline GLuint Shdr::getId()
{
    return m_id ; 
}


