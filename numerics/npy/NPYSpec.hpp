#pragma once

#include <string>
#include <climits>

#include "NPYBase.hpp"


class NPYSpec {
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


inline NPYSpec::NPYSpec(unsigned int ni, unsigned int nj, unsigned int nk, unsigned int nl, NPYBase::Type_t type)
  :
    m_ni(ni),
    m_nj(nj),
    m_nk(nk),
    m_nl(nl),
    m_bad_index(UINT_MAX), 
    m_type(type)
{
}

inline NPYBase::Type_t NPYSpec::getType()
{ 
    return m_type ;
}



 
