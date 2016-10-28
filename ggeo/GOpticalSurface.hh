#pragma once

#include <string>
struct guint4 ; 

#include "GGEO_API_EXPORT.hh"

class GGEO_API GOpticalSurface {
  public:
      static const char* dielectric_dielectric_ ;
      static const char* dielectric_metal_      ;
      static const char* Type(unsigned type);
  public:
      static const char* polished_ ;
      static const char* polishedfrontpainted_ ;
      static const char* polishedbackpainted_  ;
      static const char* ground_ ;
      static const char* groundfrontpainted_ ;
      static const char* groundbackpainted_  ;
      static const char* Finish(unsigned finish);
  public:
      static std::string brief(const guint4& optical); 
  public:
      static GOpticalSurface* create(const char* name, guint4 opt );
      GOpticalSurface(GOpticalSurface* other);
      GOpticalSurface(const char* name, const char* type, const char* model, const char* finish, const char* value);
      virtual ~GOpticalSurface();

      guint4 getOptical();
      bool isSpecular();
       

      std::string description();
      void Summary(const char* msg="GOpticalSurface::Summary", unsigned int imod=1);

      char* digest();

  private:
      void findShortName(char marker='_');

  public:
      char* getName();
      char* getType();
      char* getModel();
      char* getFinish();
      char* getValue();
      char* getShortName();

  private:
      char* m_name ;  
      char* m_type ;  
      char* m_model ;  
      char* m_finish ;  
      char* m_value ;  
      char* m_shortname ;  

};


