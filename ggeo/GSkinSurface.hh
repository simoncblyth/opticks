#pragma once


#include <string>

class GOpticalSurface ;

#include "GPropertyMap.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GSkinSurface : public GPropertyMap<float> {
  public:
      GSkinSurface(const char* name, unsigned int index, GOpticalSurface* optical_surface);
      virtual ~GSkinSurface();
      void Summary(const char* msg="GSkinSurface::Summary", unsigned int imod=1) const ;
      std::string description() const ;
  private:
      void init();
  public:
      void setSkinSurface(const char* vol);

      const char* getSkinSurfaceVol() const ;
      bool matches(const char* lv) const ;
  private:
      const char*  m_skinsurface_vol ;  

};

#include "GGEO_TAIL.hh"



