#pragma once

#include <vector>
#include "RendererBase.hh"

class NPY ;
class VecNPY ;
class Composition ; 

class Rdr : public RendererBase  {
  public:
      Rdr(const char* tag);

      enum Attrib_IDs { 
        vRdrPosition=10
      };

  public: 
      void render(unsigned int count=0, unsigned int first=0);
      void update_uniforms();
      void upload(VecNPY* vnpy, bool debug=false);
      void upload(NPY* npy, unsigned int j, unsigned int k );
      void upload(void* data, unsigned int nbytes, unsigned int stride, unsigned long offset, unsigned int countdefault);

      void setComposition(Composition* composition);
      Composition* getComposition(); 

      void setCountDefault(unsigned int countdefault);
      unsigned int getCountDefault();

  public: 
      static const char* PRINT ; 
      void configureI(const char* name, std::vector<int> values);
      void Print(const char* msg);
 

  private:
      GLuint m_vao ; 
      GLuint m_buffer ;
      unsigned int m_countdefault ; 
      Composition* m_composition ;

      GLint  m_mv_location ;
      GLint  m_mvp_location ;



};      



 
