#pragma once

#include <vector>
#include "RendererBase.hh"

class Rdr : public RendererBase  {
  public:
      Rdr(const char* tag);

      enum Attrib_IDs { 
        vRdrPosition=10
      };

  public: 
      void render(unsigned int count, unsigned int first=0);
      void upload(void* data, unsigned int nbytes, unsigned int stride, unsigned long offset );

  public: 
      static const char* PRINT ; 
      void configureI(const char* name, std::vector<int> values);
      void Print(const char* msg);

  private:
      GLuint m_vao ; 
      GLuint m_buffer ;


};      



 
