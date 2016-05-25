#pragma once

#include <vector>
#include "RendererBase.hh"
#include "NPY.hpp"

class MultiViewNPY ;
class ViewNPY ;
class NumpyEvt ; 

class Composition ; 
class Device ; 


class Rdr : public RendererBase  {
  public:
      Rdr(Device* dev, const char* tag, const char* dir=NULL, const char* incl_path=NULL);

  public: 
      void render(unsigned int count=0, unsigned int first=0);
      void check_uniforms();
      void update_uniforms();
      void dump_uniforms();

      void log(const char* msg, int value);
      void prepare_vao();


      typedef enum { LINES, LINE_STRIP, POINTS } Primitive_t ; 
      void setPrimitive( Primitive_t prim );

  public: 
      void upload(MultiViewNPY* mvn, bool debug=false);
      void upload_colors();
  private:
      void upload(NPYBase* npy, ViewNPY* vnpy);
      void attach(GLuint buffer_id);
      //void upload(void* data, unsigned int nbytes);

  public:
      // *download* : when an OpenGL buffer object is associated, glMapBuffer and read data from GPU into NPY instance 
      template <typename T>
      static void download(NPY<T>* npy);  

      static void download( NumpyEvt* evt );

      static void* mapbuffer( int buffer_id, GLenum target );
      static void unmapbuffer(GLenum target);

  private:
      //unsigned int getBufferId();

  private:
      void address(ViewNPY* vnpy);

  public: 
      static const char* PRINT ; 
      void configureI(const char* name, std::vector<int> values);
      void Print(const char* msg);

      void setComposition(Composition* composition);
      Composition* getComposition(); 

      void setCountDefault(unsigned int countdefault);
      unsigned int getCountDefault();
      
  private:
      bool    m_first_upload ; 
      Device* m_device ; 
      GLuint m_vao ; 
      bool   m_vao_generated ;
      //GLuint m_buffer ;
      unsigned int m_countdefault ; 
      Composition* m_composition ;

      // hmm consider splitting of uniform handling into 
      // separate class to avoid dupe between Renderer and Rdr ?

      GLint  m_mv_location ;
      GLint  m_mvp_location ;
      GLint  m_p_location ;
      GLint  m_isnorm_mvp_location ;
      GLint  m_selection_location ;
      GLint  m_flags_location ;
      GLint  m_pick_location ;
      GLint  m_param_location ;
      GLint  m_nrmparam_location ;
      GLint  m_scanparam_location ;
      GLint  m_timedomain_location ;
      GLint  m_colordomain_location ;
      GLint  m_colors_location ;
      GLint  m_recselect_location ;
      GLint  m_colorparam_location ;
      GLint  m_lightposition_location ;
      GLint  m_pickphoton_location ;

      GLenum m_primitive ; 

};      


inline Rdr::Rdr(Device* device, const char* tag, const char* dir, const char* incl_path)
    :
    RendererBase(tag, dir, incl_path),  
    m_first_upload(true),
    m_device(device),
    m_vao(0),
    m_vao_generated(false),
    //m_buffer(0),
    m_countdefault(0),
    m_composition(NULL),
    m_mv_location(-1),
    m_mvp_location(-1),
    m_p_location(-1),
    m_isnorm_mvp_location(-1),
    m_selection_location(-1),
    m_flags_location(-1),
    m_pick_location(-1),
    m_param_location(-1),
    m_nrmparam_location(-1),
    m_scanparam_location(-1),
    m_timedomain_location(-1),
    m_colordomain_location(-1),
    m_colors_location(-1),
    m_recselect_location(-1),
    m_colorparam_location(-1),
    m_lightposition_location(-1),
    m_pickphoton_location(-1),
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



 
