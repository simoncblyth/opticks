#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>

class Shdr {

   //friend class Prog ; 

   public:
       // ctor only reads the file, no context needed
       Shdr(const char* path, GLenum type, bool live=false);
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

};


inline GLuint Shdr::getId()
{
    return m_id ; 
}

