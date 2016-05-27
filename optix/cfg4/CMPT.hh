#pragma once

#include <vector>
#include <string>
class G4MaterialPropertiesTable ;

class CMPT {
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

inline CMPT::CMPT(G4MaterialPropertiesTable* mpt)
    :
     m_mpt(mpt)
{
}



