#pragma once

class Device ; 
template <typename T> class NPY ; 


class Colors {
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

inline Colors::Colors(Device* device)
    :
    m_device(device),
    m_colors_tex(0),
    m_colors_uploaded(false),
    m_colorbuffer(NULL)
{
}


inline void Colors::setColorBuffer(NPY<unsigned char>* colorbuffer)
{
    m_colorbuffer = colorbuffer ; 
}
