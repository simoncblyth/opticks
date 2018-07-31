#pragma once

#include <string>

#include "NPYBase.hpp"
#include "NPY_API_EXPORT.hh"

// TODO: maybe type enum and statics should live in here to simplify NPYBase

class NPY_API NPYSpec {
   public:
        NPYSpec(const char* name, unsigned int ni, unsigned int nj, unsigned int nk, unsigned int nl, unsigned int nm, NPYBase::Type_t type, const char* ctrl, bool optional=false);
        void setNumItems(unsigned ni) ; 

        NPYSpec* clone() const ; 
        NPYBase::Type_t getType() const ;

        const char*     getName() const ;
        const char*     getTypeName() const ;
        const char*     getCtrl() const ;
        unsigned int    getDimension(unsigned int i) const ;
        bool isOptional() const ; 
        bool isEqualTo(const NPYSpec* other) const ;
        std::string     description() const  ;
        std::string     desc() const  ;
        void Summary(const char* msg="NPYSpec::Summary") const ;

   private:
        const char*      m_name ; 
        unsigned         m_ni ; 
        unsigned         m_nj ; 
        unsigned         m_nk ; 
        unsigned         m_nl ; 
        unsigned         m_nm ; 
        unsigned int     m_bad_index ; 
        NPYBase::Type_t  m_type ; 
        const char*      m_ctrl  ; 
        bool             m_optional ; 
};


 
