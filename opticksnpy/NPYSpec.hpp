#pragma once

#include <string>

#include "NPYBase.hpp"
#include "NPY_API_EXPORT.hh"

// TODO: maybe type enum and statics should live in here to simplify NPYBase

class NPY_API NPYSpec {
   public:
        NPYSpec(const char* name, unsigned int ni, unsigned int nj, unsigned int nk, unsigned int nl, unsigned int nm, NPYBase::Type_t type, const char* ctrl);

        NPYSpec* clone(); 

        NPYBase::Type_t getType();
        const char* getName();
        const char* getTypeName();
        const char* getCtrl();
        unsigned int getDimension(unsigned int i) ;
        bool isEqualTo(NPYSpec* other) ;

        std::string description() ;
        void Summary(const char* msg="NPYSpec::Summary") ;
   private:
        const char*  m_name ; 
        unsigned int m_ni ; 
        unsigned int m_nj ; 
        unsigned int m_nk ; 
        unsigned int m_nl ; 
        unsigned int m_nm ; 

        unsigned int m_bad_index ; 
        NPYBase::Type_t  m_type ; 
        const char*  m_ctrl  ; 
};


 
