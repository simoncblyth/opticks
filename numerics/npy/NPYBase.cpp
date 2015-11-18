#include "NPYBase.hpp"

#include "string.h"
#include <sstream>
#include <iostream>

// npy-
#include "stringutil.hpp"
#include "md5digest.hpp"

//bregex- 
#include "regexsearch.hh"

#include <boost/algorithm/string/replace.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

const char* NPYBase::DEFAULT_PATH_TEMPLATE = "$LOCAL_BASE/env/$1/$2/%s.npy" ; 


void NPYBase::setNumItems(unsigned int ni)
{
    unsigned int orig = m_shape[0] ;
    assert(ni >= orig);

    LOG(debug) << "NPYBase::setNumItems"
              << " increase from " << orig << " to " << ni 
              ; 
 
    m_shape[0] = ni ; 
    m_len0 = getShape(0);
}


std::string NPYBase::getDigestString()
{
    return getDigestString(getBytes(), getNumBytes(0));
}

std::string NPYBase::getDigestString(void* bytes, unsigned int nbytes)
{
    MD5Digest dig ;
    dig.update( (char*)bytes, nbytes);
    return dig.finalize();
}

bool NPYBase::isEqualTo(NPYBase* other)
{
    return isEqualTo(other->getBytes(), other->getNumBytes(0));
}

bool NPYBase::isEqualTo(void* bytes, unsigned int nbytes)
{
    std::string self = getDigestString();
    std::string other = getDigestString(bytes, nbytes);

    bool same = self.compare(other) == 0 ; 

    if(!same)
         LOG(warning) << "NPYBase::isEqualTo NO "
                      << " self " << self 
                      << " other " << other
                      ;
 

    return same ; 
}




std::string NPYBase::getShapeString(unsigned int ifr)
{
    return getItemShape(ifr);
}

std::string NPYBase::getItemShape(unsigned int ifr)
{
    std::stringstream ss ; 
    for(size_t i=ifr ; i < m_shape.size() ; i++)
    {
        ss << m_shape[i]  ;
        if( i < m_shape.size() - 1) ss << "," ;
    }
    return ss.str(); 
}


void NPYBase::Summary(const char* msg)
{
    std::string desc = description(msg);
    LOG(info) << desc ; 
}   













std::string NPYBase::description(const char* msg)
{
    std::stringstream ss ; 

    ss << msg << " (" ;

    for(size_t i=0 ; i < m_shape.size() ; i++)
    {
        ss << m_shape[i]  ;
        if( i < m_shape.size() - 1) ss << "," ;
    }
    ss << ") " ;
    ss << " len0 " << m_len0 ;
    ss << " len1 " << m_len1 ;
    ss << " len2 " << m_len2 ;

    ss << " getNumBytes(0) " << getNumBytes(0) ;
    ss << " getNumBytes(1) " << getNumBytes(1) ;
    ss << " getNumValues(0) " << getNumValues(0) ;
    ss << " getNumValues(1) " << getNumValues(1) ;

    ss << m_metadata  ;

    return ss.str();
}

std::string NPYBase::path(const char* pfx, const char* gen, const char* tag, const char* det)
{
    std::stringstream ss ;
    ss << pfx << gen ;
    return path(ss.str().c_str(), tag, det );
}
  
std::string NPYBase::path(const char* typ, const char* tag, const char* det)
{
/*
:param typ: object type name, eg oxcerenkov rxcerenkov 
:param tag: event tag, usually numerical 
:param det: detector tag, eg dyb, juno

The typ is used to identify the name of an envvar 
in which the template path at which to save/load such 
objects must be found.

For example for typ "rxcerenkov" the  below envvar
must be present in the environment. 

    DAE_RXCERENKOV_PATH_TEMPLATE 

Envvars are defined in env/export-

*/


   // const char* TYP = uppercase(typ);
   // char envvar[64];
   // snprintf(envvar, 64, "DAE_%s_PATH_TEMPLATE", TYP ); 
   // free((void*)TYP); 


    char path_[256];

    std::string deftmpl(DEFAULT_PATH_TEMPLATE) ; 
    //char* tmpl = getenv(envvar) ;
    //if(!tmpl)
    //{

        boost::replace_first(deftmpl, "$1", det );
        boost::replace_first(deftmpl, "$2", typ );
        deftmpl = os_path_expandvars( deftmpl.c_str() ); 
        char* tmpl = (char*)deftmpl.c_str();

       //   LOG(debug)<<"NPY<T>::path using default path template " << tmpl  
       //               << " as envvar " << envvar << " is  not defined "  
       //               << " define new typs with env-;export-;export-vi " ; 
        // LOG(fatal)<< "NPY<T>::path missing envvar for "
        //           << " typ " << typ 
        //           << " envvar " << envvar  
        // assert(0);
        // return "missing-template-envvar" ; 
    //}
    snprintf(path_, 256, tmpl, tag );
    return path_ ;   
}



std::string NPYBase::path(const char* dir, const char* name)
{
    char path[256];
    snprintf(path, 256, "%s/%s", dir, name);
    return path ; 
}




