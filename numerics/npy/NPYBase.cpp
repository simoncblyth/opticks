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


#include "NLog.hpp"


const char* NPYBase::DEFAULT_PATH_TEMPLATE = "$LOCAL_BASE/env/opticks/$1/$2/%s.npy" ; 

void NPYBase::setNumItems(unsigned int ni)
{
    unsigned int orig = m_shape[0] ;
    assert(ni >= orig);

    LOG(debug) << "NPYBase::setNumItems"
              << " increase from " << orig << " to " << ni 
              ; 
 
    setShape(0, ni);
}


void NPYBase::reshape(int ni, unsigned int nj, unsigned int nk, unsigned int nl)
{
    unsigned int nv_old = m_ni*m_nj*m_nk*m_nl ; 
    if(ni < 0) ni = nv_old/(nj*nk*nl) ;

    unsigned int nv_new = ni*nj*nk*nl ; 

    if(nv_old != nv_new) LOG(fatal) << "NPYBase::reshape INVALID AS CHANGES COUNTS " 
                              << " nv_old " << nv_old 
                              << " nv_new " << nv_new
                              ;

    assert(nv_old != nv_new && "NPYBase::reshape cannot change number of values, just their addressing");

    setShape(0, ni);
    setShape(1, nj);
    setShape(2, nk);
    setShape(3, nl);
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
    ss << " ni " << m_ni ;
    ss << " nj " << m_nj ;
    ss << " nk " << m_nk ;
    ss << " nl " << m_nl ;

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

    char path_[256];

    std::string deftmpl(DEFAULT_PATH_TEMPLATE) ; 

    boost::replace_first(deftmpl, "$1", det );
    boost::replace_first(deftmpl, "$2", typ );
    deftmpl = os_path_expandvars( deftmpl.c_str() ); 
    char* tmpl = (char*)deftmpl.c_str();

    snprintf(path_, 256, tmpl, tag );
    return path_ ;   
}



std::string NPYBase::path(const char* dir, const char* name)
{
    char path[256];
    snprintf(path, 256, "%s/%s", dir, name);
    return path ; 
}




