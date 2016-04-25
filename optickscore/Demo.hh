#pragma once

#include <vector>
#include <string>

#include "NConfigurable.hpp"

class Demo : public NConfigurable  {
   public:
       static const char* A ;
       static const char* B ;
       static const char* C ;
       static const char* PREFIX ;
       const char* getPrefix();
   public:
       Demo();
   public:
     // Configurable
     std::vector<std::string> getTags();
     void set(const char* name, std::string& xyz);
     std::string get(const char* name);
   public:
       float getA();
       float getB();
       float getC();

       void setA(float a);
       void setB(float b);
       void setC(float c);

   private:
       float m_a ; 
       float m_b ; 
       float m_c ; 

};


inline Demo::Demo() 
  :
   m_a(0.f),
   m_b(0.f),
   m_c(0.f)
{
}

inline float Demo::getA(){ return m_a ; }
inline float Demo::getB(){ return m_b ; }
inline float Demo::getC(){ return m_c ; }

inline void Demo::setA(float a){ m_a = a ; }
inline void Demo::setB(float b){ m_b = b ; }
inline void Demo::setC(float c){ m_c = c ; }

inline std::vector<std::string> Demo::getTags()
{
    std::vector<std::string> tags ;
    tags.push_back(A);
    tags.push_back(B);
    //tags.push_back(C);
    return tags ; 
}


