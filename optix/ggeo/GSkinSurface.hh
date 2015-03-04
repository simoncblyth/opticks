#ifndef GSKINSURFACE_H
#define GSKINSURFACE_H

#include "GPropertyMap.hh"

class GSkinSurface : public GPropertyMap {
  public:
      GSkinSurface(const char* name, unsigned int index);
      virtual ~GSkinSurface();
      void Summary(const char* msg="GSkinSurface::Summary");

  public:
      void setSkinSurface(const char* vol);
      char* getSkinSurfaceVol();
      bool matches(const char* lv);

  private:
      char* m_skinsurface_vol ;  


};

#endif


