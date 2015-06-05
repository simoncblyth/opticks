#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>

class Shdr {

   //friend class Prog ; 

   public:
       static const char* include_prefix ; 
       static const char* enum_prefix ; 

       // ctor only reads the file, no context needed
       Shdr(const char* path, GLenum type, bool live=false);

    public:
       void setIncludePath(const char* path); // colon delimited list of directories to look for glsl inclusions
       std::string resolve(const char* name);

       void createAndCompile();
       void Print(const char* msg);
       GLuint getId();
 
   private:
       void readFile(const char* path);
       void _print_shader_info_log();

   private:
       char*      m_path ; 
       GLenum     m_type ; 
       GLuint     m_id ;
       bool       m_live ;

       std::string              m_content;
       std::vector<std::string> m_lines ; 

       std::string              m_include_path ; 
       std::vector<std::string> m_include_dirs ; 

};


inline GLuint Shdr::getId()
{
    return m_id ; 
}


