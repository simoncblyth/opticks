#pragma once
#include <string>
#include <cassert>

class GOpticalSurface {
  public:
      GOpticalSurface(GOpticalSurface* other);
      GOpticalSurface(const char* name, const char* type, const char* model, const char* finish, const char* value);
      virtual ~GOpticalSurface();

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

inline char* GOpticalSurface::getName()
{
    return m_name ; 
}
inline char* GOpticalSurface::getType()
{
    return m_type ; 
}
inline char* GOpticalSurface::getModel()
{
    return m_model ; 
}
inline char* GOpticalSurface::getFinish()
{
    return m_finish ; 
}

/*
 source/materials/include/G4OpticalSurface.hh 

 61 enum G4OpticalSurfaceFinish
 62 {
 63    polished,                    // smooth perfectly polished surface
 64    polishedfrontpainted,        // smooth top-layer (front) paint
 65    polishedbackpainted,         // same is 'polished' but with a back-paint
 66 
 67    ground,                      // rough surface
 68    groundfrontpainted,          // rough top-layer (front) paint
 69    groundbackpainted,           // same as 'ground' but with a back-paint
 70 

*/




inline char* GOpticalSurface::getValue()
{
    return m_value ; 
}

inline char* GOpticalSurface::getShortName()
{
    return m_shortname ; 
}


