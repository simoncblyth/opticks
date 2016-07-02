#pragma once

#include <vector>
#include <string>
class G4MaterialPropertiesTable ;

#include "CFG4_API_EXPORT.hh"
class CFG4_API CMPT {
   public:
       CMPT(G4MaterialPropertiesTable* mpt);
   public:
       std::string description(const char* msg);
       std::vector<std::string> getPropertyKeys();
       std::vector<std::string> getPropertyDesc();
       std::vector<std::string> getConstPropertyKeys();
       std::vector<double> getConstPropertyValues();
   private:
       G4MaterialPropertiesTable* m_mpt ; 
};



