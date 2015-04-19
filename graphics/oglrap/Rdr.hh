#pragma once

#include <vector>
#include "RendererBase.hh"

class NPY ;
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

  public: 
      void upload(MultiVecNPY* mvn);
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



};      

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



 
