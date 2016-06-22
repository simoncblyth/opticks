#pragma once

#include "GPropertyMap.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GSource : public GPropertyMap<float> {
  public:
      static GSource* make_blackbody_source(const char* name, unsigned int index, float kelvin=6500.f);
  public:
      GSource(GSource* other);
      GSource(const char* name, unsigned int index);
      virtual ~GSource();
  private:
      void init(); 
  public: 
      void Summary(const char* msg="GSource::Summary");

};

#include "GGEO_TAIL.hh"




