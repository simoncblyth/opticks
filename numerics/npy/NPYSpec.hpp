#pragma once

#include <string>

#include "NPYBase.hpp"
#include "NPY_API_EXPORT.hh"

// TODO: maybe type enum and statics should live in here to simplify NPYBase

class NPY_API NPYSpec {
   public:
        NPYSpec(unsigned int ni, unsigned int nj, unsigned int nk, unsigned int nl, NPYBase::Type_t type);

        NPYBase::Type_t getType();
        const char* getTypeName();
        unsigned int getDimension(unsigned int i) ;
        bool isEqualTo(NPYSpec* other) ;

        std::string description() ;
        void Summary(const char* msg="NPYSpec::Summary") ;
   private:
        unsigned int m_ni ; 
        unsigned int m_nj ; 
        unsigned int m_nk ; 
        unsigned int m_nl ; 
        unsigned int m_bad_index ; 
        NPYBase::Type_t  m_type ; 
};


 
