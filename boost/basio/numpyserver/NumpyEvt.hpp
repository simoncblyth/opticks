#pragma once

#include <string>
#include <sstream>


#include "NPY.hpp"

class NumpyEvt {
   public:
       NumpyEvt();

       void setNPY(NPY* npy);
       NPY* getNPY();
       bool hasNPY();

       std::string description(const char* msg);

   private:
       NPY* m_npy ;
};

NumpyEvt::NumpyEvt() :
   m_npy(NULL)
{
}

void NumpyEvt::setNPY(NPY* npy)
{
    m_npy = npy ;
}

NPY* NumpyEvt::getNPY()
{
    return m_npy ;
}

bool NumpyEvt::hasNPY()
{
    return m_npy != NULL ;
}


std::string NumpyEvt::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " " ;
    if(m_npy)
    {
         ss << m_npy->description("m_npy") ;
    }
    else
    {
         ss << "empty" ;
    }
    return ss.str();
}




