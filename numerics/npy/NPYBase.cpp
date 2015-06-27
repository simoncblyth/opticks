#include "NPYBase.hpp"

#include "string.h"
#include <sstream>
#include <iostream>

// npy-
#include "stringutil.hpp"

//bregex- 
#include "regexsearch.hh"

#include <boost/algorithm/string/replace.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

const char* NPYBase::DEFAULT_PATH_TEMPLATE = "$LOCAL_BASE/env/$1/%s.npy" ; 


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
    std::cout << desc << std::endl ; 
}   

void NPYBase::dump(const char* msg)
{
    LOG(info) << msg ; 
    std::string desc = description(msg);
    std::cout << desc << std::endl ; 
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


std::string NPYBase::path(const char* typ, const char* tag)
{
/*
:param typ: object type name, eg oxcerenkov rxcerenkov 
:param tag: event tag, usually numerical 

The typ is used to identify the name of an envvar 
in which the template path at which to save/load such 
objects must be found.

For example for typ "rxcerenkov" the  below envvar
must be present in the environment. 

    DAE_RXCERENKOV_PATH_TEMPLATE 

Envvars are defined in env/export-

*/
    const char* TYP = uppercase(typ);
    char envvar[64];
    snprintf(envvar, 64, "DAE_%s_PATH_TEMPLATE", TYP ); 
    free((void*)TYP); 


    char path_[256];

    std::string deftmpl(DEFAULT_PATH_TEMPLATE) ; 

    char* tmpl = getenv(envvar) ;
    if(!tmpl)
    {
        boost::replace_first(deftmpl, "$1", typ );
        deftmpl = os_path_expandvars( deftmpl.c_str() ); 
        tmpl = (char*)deftmpl.c_str();

        LOG(warning)<<"NPY<T>::path using default path template " << tmpl  
                    << " as envvar " << envvar << " is  not defined "  
                    << " define new typs with env-;export-;export-vi " ; 

        // LOG(fatal)<< "NPY<T>::path missing envvar for "
        //           << " typ " << typ 
        //           << " envvar " << envvar  
        // assert(0);
        // return "missing-template-envvar" ; 
    }
    snprintf(path_, 256, tmpl, tag );
    return path_ ;   
}







