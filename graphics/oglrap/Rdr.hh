#pragma once

#include <vector>
#include "RendererBase.hh"
#include "NPY.hpp"

class MultiVecNPY ;
class VecNPY ;
class Composition ; 

class Rdr : public RendererBase  {
  public:
      Rdr(const char* tag);

  public: 
      void render(unsigned int count=0, unsigned int first=0);
      void check_uniforms();
      void update_uniforms();
      void dump_uniforms();

      typedef enum { LINES, LINE_STRIP, POINTS } Primitive_t ; 
      void setPrimitive( Primitive_t prim );

  public: 
      void upload(MultiVecNPY* mvn);

      // *download* : when an OpenGL buffer object is associated, glMapBuffer and read data from GPU into NPY instance 
      template <typename T>
      static void download(NPY<T>* npy);  

      static void* mapbuffer( int buffer_id, GLenum target );
      static void unmapbuffer(GLenum target);
      unsigned int getBufferId();

  private:
      void address(VecNPY* vnpy);
      void upload(void* data, unsigned int nbytes);

  public: 
      static const char* PRINT ; 
      void configureI(const char* name, std::vector<int> values);
      void Print(const char* msg);

      void setComposition(Composition* composition);
      Composition* getComposition(); 

      void setCountDefault(unsigned int countdefault);
      unsigned int getCountDefault();
      
  private:
      GLuint m_vao ; 
      GLuint m_buffer ;
      unsigned int m_countdefault ; 
      Composition* m_composition ;

      GLint  m_mv_location ;
      GLint  m_mvp_location ;
      GLint  m_isnorm_mvp_location ;
      GLint  m_selection_location ;
      GLint  m_flags_location ;
      GLint  m_param_location ;
      GLint  m_timedomain_location ;

      GLenum m_primitive ; 

};      


inline Rdr::Rdr(const char* tag)
    :
    RendererBase(tag),
    m_vao(0),
    m_buffer(0),
    m_countdefault(0),
    m_composition(NULL),
    m_mv_location(-1),
    m_mvp_location(-1),
    m_isnorm_mvp_location(-1),
    m_selection_location(-1),
    m_flags_location(-1),
    m_param_location(-1),
    m_timedomain_location(-1),
    m_primitive(GL_POINTS)
{
}


template <typename T>
inline void Rdr::download( NPY<T>* npy )
{
    GLenum target = GL_ARRAY_BUFFER ;
    void* ptr = mapbuffer( npy->getBufferId(), target );
    if(ptr)
    {
       npy->read(ptr);
       unmapbuffer(target);
    }
}


inline unsigned int Rdr::getBufferId()
{
   return m_buffer ; 
}

inline void Rdr::configureI(const char* name, std::vector<int> values )
{
    if(values.empty()) return ; 
    if(strcmp(name, PRINT)==0) Print("Rdr::configureI");
}
inline void Rdr::Print(const char* msg)
{
    printf("%s\n", msg);
}
inline void Rdr::setCountDefault(unsigned int count)
{
    m_countdefault = count ;
}
inline unsigned int Rdr::getCountDefault()
{
    return m_countdefault ;
}
inline void Rdr::setComposition(Composition* composition)
{
    m_composition = composition ;
}
inline Composition* Rdr::getComposition()
{
    return m_composition ;
}



 
