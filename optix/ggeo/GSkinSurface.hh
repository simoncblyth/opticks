#pragma once
#include "GPropertyMap.hh"

class GOpticalSurface ;

class GSkinSurface : public GPropertyMap<float> {
  public:
      GSkinSurface(const char* name, unsigned int index, GOpticalSurface* optical_surface);
      virtual ~GSkinSurface();
      void Summary(const char* msg="GSkinSurface::Summary", unsigned int imod=1);

  public:
      GOpticalSurface* getOpticalSurface();
  private:
      void setOpticalSurface(GOpticalSurface* os);

  public:
      void setSkinSurface(const char* vol);
      char* getSkinSurfaceVol();
      bool matches(const char* lv);

  private:
      char*              m_skinsurface_vol ;  
      GOpticalSurface*   m_optical_surface ; 


};


inline GOpticalSurface* GSkinSurface::getOpticalSurface()
{
    return m_optical_surface ; 
}
inline void GSkinSurface::setOpticalSurface(GOpticalSurface* optical_surface)
{
    m_optical_surface = optical_surface ; 
}





