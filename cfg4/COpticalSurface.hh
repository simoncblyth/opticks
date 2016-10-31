#pragma once

#include <string>
class G4OpticalSurface ; 

#include "CFG4_API_EXPORT.hh"
class CFG4_API COpticalSurface 
{
  public:
      static std::string Brief(G4OpticalSurface* os);
  public:
      static const char* dielectric_dielectric_ ;
      static const char* dielectric_metal_      ;
      static const char* Type(G4SurfaceType type);
  public:
      static const char* polished_ ;
      static const char* polishedfrontpainted_ ;
      static const char* polishedbackpainted_  ;
      static const char* ground_ ;
      static const char* groundfrontpainted_ ;
      static const char* groundbackpainted_  ;
      static const char* Finish( G4OpticalSurfaceFinish finish);
  public:
      static const char* glisur_ ;
      static const char* unified_;
      static const char* Model( G4OpticalSurfaceModel model );
  public:
      COpticalSurface(G4OpticalSurface* os);
      std::string brief();
  private:
      G4OpticalSurface* m_os ; 

};
