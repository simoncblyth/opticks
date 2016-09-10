#pragma once

#include <string>
#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OEntry 
{ 
   public:
        OEntry(unsigned index, char code);
   public:
        unsigned    getIndex();
        std::string description();
        bool        isTrivial();
   private:
        unsigned             m_index ; 
        char                 m_code ; 

};
