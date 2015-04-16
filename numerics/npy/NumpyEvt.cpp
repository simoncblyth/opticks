#include "NumpyEvt.hpp"
#include "NPY.hpp"
#include <sstream>

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



