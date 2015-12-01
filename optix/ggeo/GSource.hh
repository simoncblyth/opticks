#pragma once

#include "GPropertyMap.hh"

class GSource : public GPropertyMap<float> {
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




