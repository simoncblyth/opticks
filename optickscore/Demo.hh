#pragma once

#include <vector>
#include <string>

#include "OKCORE_API_EXPORT.hh"
#include "NConfigurable.hpp"

class OKCORE_API Demo : public NConfigurable  {
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


