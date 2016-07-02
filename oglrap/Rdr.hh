#pragma once

#include <vector>

template <typename T> class NPY ; 
class MultiViewNPY ;
class ViewNPY ;

class OpticksEvent ; 
class Composition ; 

class Device ; 

#include "RendererBase.hh"
#include "OGLRAP_API_EXPORT.hh"
#include "OGLRAP_HEAD.hh"

class OGLRAP_API Rdr : public RendererBase  {
  public:
      Rdr(Device* dev, const char* tag, const char* dir=NULL, const char* incl_path=NULL);

  public: 
      void render(unsigned int count=0, unsigned int first=0);
      void check_uniforms();
      void update_uniforms();
      void dump_uniforms();
      void dump_uploads_table(const char* msg="Rdr::dump_uploads_table");

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

      static void download( OpticksEvent* evt );

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

      std::vector<MultiViewNPY*> m_uploads ; 
};      

#include "OGLRAP_TAIL.hh"

 
