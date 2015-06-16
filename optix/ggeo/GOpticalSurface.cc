#include "GOpticalSurface.hh"

#include "stdlib.h"
#include "string.h"
#include "stdio.h"

GOpticalSurface::GOpticalSurface(const char* name, const char* type, const char* model, const char* finish, const char* value) 
    : 
    m_name(strdup(name)), 
    m_type(strdup(type)), 
    m_model(strdup(model)), 
    m_finish(strdup(finish)), 
    m_value(strdup(value)) 
{
}

GOpticalSurface::GOpticalSurface(GOpticalSurface* other)
   :
   m_name(strdup(other->getName())),
   m_type(strdup(other->getType())),
   m_model(strdup(other->getModel())),
   m_finish(strdup(other->getFinish())),
   m_value(strdup(other->getValue()))
{
} 


GOpticalSurface::~GOpticalSurface()
{
    free(m_name);
    free(m_type);
    free(m_model);
    free(m_finish);
    free(m_value);
}

void GOpticalSurface::Summary(const char* msg, unsigned int imod)
{
    printf("%s\n", msg );
    printf(" name %s type %s model %s finish %s value %s \n", m_name, m_type, m_model, m_finish, m_value );
}




