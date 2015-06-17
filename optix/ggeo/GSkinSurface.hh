#pragma once
#include "GPropertyMap.hh"
#include <string>

class GOpticalSurface ;

class GSkinSurface : public GPropertyMap<float> {
  public:
      GSkinSurface(const char* name, unsigned int index, GOpticalSurface* optical_surface);
      virtual ~GSkinSurface();
      void Summary(const char* msg="GSkinSurface::Summary", unsigned int imod=1);

  public:
      std::string description();

  public:
      void setSkinSurface(const char* vol);
      char* getSkinSurfaceVol();
      bool matches(const char* lv);

  private:
      char*              m_skinsurface_vol ;  

};




