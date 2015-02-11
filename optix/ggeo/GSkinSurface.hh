#ifndef GSKINSURFACE_H
#define GSKINSURFACE_H

#include "GPropertyMap.hh"

class GSkinSurface : public GPropertyMap {
  public:
      GSkinSurface(const char* name);
      virtual ~GSkinSurface();
      void Summary(const char* msg="GSkinSurface::Summary");

  public:
      void setSkinSurface(const char* vol);
      char* getSkinSurfaceVol();

  private:
      char* m_skinsurface_vol ;  


};

#endif


