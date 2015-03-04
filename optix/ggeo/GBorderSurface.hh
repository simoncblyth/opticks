#ifndef GBORDERSURFACE_H
#define GBORDERSURFACE_H

#include "GPropertyMap.hh"

class GBorderSurface : public GPropertyMap {
  public:
      GBorderSurface(const char* name, unsigned int index);
      virtual ~GBorderSurface();
      void Summary(const char* msg="GBorderSurface::Summary");

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


};

#endif


