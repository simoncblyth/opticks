#include "GOpticalSurface.hh"

#include "md5digest.hpp"
#include "stdlib.h"
#include "string.h"
#include "stdio.h"
#include <sstream>
#include <iomanip>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



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

    // dyb names start /dd/... which is translated to __dd__
    // so detect this and apply the shortening
    // 
    // juno names do not have the prefix so make shortname
    // the same as the full one
    //
    // have to have different treatment as juno has multiple names ending _opsurf
    // which otherwise get shortened to "opsurf" and tripup the digest checking
    //
    m_shortname = m_name[0] == marker ? strrchr(m_name, marker) + 1 : m_name ; 

    LOG(debug) << __func__
              << " name [" << m_name << "]" 
              << " shortname [" << m_shortname << "]" 
              ;

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

char* GOpticalSurface::digest()
{
    MD5Digest dig ;
    dig.update( m_type,   strlen(m_type) );
    dig.update( m_model,  strlen(m_model) );
    dig.update( m_finish, strlen(m_finish) );
    dig.update( m_value,  strlen(m_value) );
    return dig.finalize();
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
       << std::setw(30) << m_shortname 
       ;

    return ss.str();
}




