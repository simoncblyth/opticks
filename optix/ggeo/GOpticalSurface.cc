#include "GOpticalSurface.hh"

#include "stdlib.h"
#include "string.h"
#include "stdio.h"
#include <sstream>
#include <iomanip>

GOpticalSurface::GOpticalSurface(const char* name, const char* type, const char* model, const char* finish, const char* value) 
    : 
    m_name(strdup(name)), 
    m_type(strdup(type)), 
    m_model(strdup(model)), 
    m_finish(strdup(finish)), 
    m_value(strdup(value)),
    m_shortname(NULL)
{
    findShortName();
}

GOpticalSurface::GOpticalSurface(GOpticalSurface* other)
   :
   m_name(strdup(other->getName())),
   m_type(strdup(other->getType())),
   m_model(strdup(other->getModel())),
   m_finish(strdup(other->getFinish())),
   m_value(strdup(other->getValue())),
   m_shortname(NULL)
{
    findShortName();
} 


void GOpticalSurface::findShortName(char marker)
{
    if(m_shortname) return ;
    m_shortname = strrchr(m_name, marker) + 1 ; 
}


GOpticalSurface::~GOpticalSurface()
{
    free(m_name);
    free(m_type);
    free(m_model);
    free(m_finish);
    free(m_value);
    free(m_shortname);
}

void GOpticalSurface::Summary(const char* msg, unsigned int imod)
{
    printf("%s : type %s model %s finish %s value %4s shortname %s \n", msg, m_type, m_model, m_finish, m_value, m_shortname );
}

std::string GOpticalSurface::description()
{
    std::stringstream ss ; 

    ss << " GOpticalSurface " 
       << " type " << m_type 
       << " model " << m_model 
       << " finish " << m_finish
       << " value " << std::setw(5) << m_value
       << std::setw(25) << m_shortname 
       ;

    return ss.str();
}




