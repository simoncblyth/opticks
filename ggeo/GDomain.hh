#pragma once

#include <string>
#include "GGEO_API_EXPORT.hh"


template <class T>
class GGEO_API GDomain {
  public: 
     static GDomain<T>* GetDefaultDomain() ; 
     static unsigned Length(T low, T high, T step);
  private: 
     static GDomain<T>* fDefaultDomain ; 
  public: 
     GDomain(T low, T high, T step);
     virtual ~GDomain() {}
  public: 
     GDomain<T>* makeInterpolationDomain(T step) const ;
  public: 
     T getLow() const {  return m_low ; }   
     T getHigh() const { return m_high ; }   
     T getStep() const { return m_step ; }
     unsigned getLength() const { return m_length ; }
  public: 
     std::string desc() const ; 
     void Summary(const char* msg="GDomain::Summary") const ;
     bool isEqual(GDomain<T>* other) const ; 
     T* getValues() const ;   

  private:
     T m_low ; 
     T m_high ; 
     T m_step ; 
     unsigned m_length ; 


};



