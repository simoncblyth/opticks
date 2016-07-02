#pragma once

class Device ; 
template <typename T> class NPY ; 

#include "OGLRAP_API_EXPORT.hh"

class OGLRAP_API Colors {
  public:
      Colors(Device* dev);
  public: 
      void setColorBuffer(NPY<unsigned char>* colorbuffer);
      unsigned int getNumColors();
  public: 
      void upload();
  private:
      Device*      m_device ; 
      GLuint       m_colors_tex ;
      bool         m_colors_uploaded ; 
      NPY<unsigned char>*     m_colorbuffer ; 


};      


