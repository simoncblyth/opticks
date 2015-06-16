#pragma once

class GOpticalSurface {
  public:
      GOpticalSurface(GOpticalSurface* other);
      GOpticalSurface(const char* name, const char* type, const char* model, const char* finish, const char* value);
      virtual ~GOpticalSurface();

      void Summary(const char* msg="GOpticalSurface::Summary", unsigned int imod=1);

  public:
      char* getName();
      char* getType();
      char* getModel();
      char* getFinish();
      char* getValue();

  private:
      char* m_name ;  
      char* m_type ;  
      char* m_model ;  
      char* m_finish ;  
      char* m_value ;  

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
inline char* GOpticalSurface::getValue()
{
    return m_value ; 
}

