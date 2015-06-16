#pragma once
#include "GPropertyMap.hh"

class GOpticalSurface ; 

class GBorderSurface : public GPropertyMap<float> {
  public:
      GBorderSurface(const char* name, unsigned int index, GOpticalSurface* optical_surface );
      virtual ~GBorderSurface();
      void Summary(const char* msg="GBorderSurface::Summary", unsigned int imod=1);

  public:
      GOpticalSurface* getOpticalSurface();

  private:
      void setOpticalSurface(GOpticalSurface* os);

  public:
      void setBorderSurface(const char* pv1, const char* pv2);
      char* getBorderSurfacePV1();
      char* getBorderSurfacePV2();

  public:
      bool matches(const char* pv1, const char* pv2);
      bool matches_swapped(const char* pv1, const char* pv2);
      bool matches_either(const char* pv1, const char* pv2);
      bool matches_one(const char* pv1, const char* pv2);

  private:
      char* m_bordersurface_pv1 ;  
      char* m_bordersurface_pv2 ;  

      GOpticalSurface* m_optical_surface ; 

};


inline GOpticalSurface* GBorderSurface::getOpticalSurface()
{
    return m_optical_surface ; 
}
inline void GBorderSurface::setOpticalSurface(GOpticalSurface* optical_surface)
{
    m_optical_surface = optical_surface ; 
}



