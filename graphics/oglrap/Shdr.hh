#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>

class Shdr {
   public:
       Shdr(const char* path, GLenum type, bool live=false);
       void Print(const char* msg);
 
   private:
       void init();
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



