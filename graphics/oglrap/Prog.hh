#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>

class Prog {
   public:
      Prog(const char* basedir, const char* tag);

   private:
      void setup();
      void examine(const char* tagdir);
      void addShader(const char* path, GLenum type);

   private:
     // GLuint m_id ;
      std::vector<std::string> m_names;
      std::vector<GLenum>      m_codes;

};



