#include "NPYBase.hpp"

#include <cstring>
#include <sstream>
#include <iostream>
#include <algorithm>

//brap- 
#include "fsutil.hh"
#include "regexsearch.hh"
#include "stringutil.hh"
#include "md5digest.hh"

#include <boost/algorithm/string/replace.hpp>

#include "NPYSpec.hpp"
#include "BLog.hh"


bool NPYBase::GLOBAL_VERBOSE = false ; 


const char* NPYBase::FLOAT_ = "FLOAT" ; 
const char* NPYBase::SHORT_ = "SHORT" ; 
const char* NPYBase::DOUBLE_ = "DOUBLE" ; 
const char* NPYBase::INT_ = "INT" ; 
const char* NPYBase::UINT_ = "UINT" ; 
const char* NPYBase::CHAR_ = "CHAR" ; 
const char* NPYBase::UCHAR_ = "UCHAR" ; 
const char* NPYBase::ULONGLONG_ = "ULONGLONG" ; 

const char* NPYBase::TypeName(Type_t type)
{
    const char* name = NULL ; 
    switch(type)
    { 
        case FLOAT:name=FLOAT_;break;
        case SHORT:name=SHORT_;break;
        case DOUBLE:name=DOUBLE_;break;
        case INT:name=INT_;break;
        case UINT:name=UINT_;break;
        case CHAR:name=CHAR_;break;
        case UCHAR:name=UCHAR_;break;
        case ULONGLONG:name=ULONGLONG_;break;
    } 
    return name ; 
}




void NPYBase::setGlobalVerbose(bool verbose)
{
    GLOBAL_VERBOSE = verbose ;
}

const char* NPYBase::DEFAULT_DIR_TEMPLATE = "$LOCAL_BASE/env/opticks/$1/$2" ; 

void NPYBase::init()
{
   updateDimensions(); 
   m_shape_spec = new NPYSpec(m_ni, m_nj, m_nk, m_nl, m_type ); 
   m_item_spec = new NPYSpec(0, m_nj, m_nk, m_nl, m_type ); 
}


void NPYBase::updateDimensions()
{
    m_ni = getShape(0); 
    m_nj = getShape(1);
    m_nk = getShape(2);
    m_nl = getShape(3);  // gives 0 when beyond dimensions
    m_dim = m_shape.size();
}


unsigned int NPYBase::getNumQuads()
{
   unsigned int num_quad ;  
   unsigned int ndim = m_shape.size() ;
   unsigned int last_dimension = ndim > 1 ? m_shape[ndim-1] : 0  ;

   if(last_dimension != 4 )
   {
       LOG(warning) << "NPYBase::getNumQuads last dim expected to be 4  " << getShapeString()  ;
       num_quad = 0 ; 
   } 
   else
   {
       num_quad = 1 ; 
       for(unsigned int i=0 ; i < ndim - 1 ; i++ ) num_quad *= m_shape[i] ; 
   } 
   return num_quad ;
}



bool NPYBase::hasShape(unsigned int ni, unsigned int nj, unsigned int nk, unsigned int nl)
{
    return m_ni == ni && m_nj == nj && m_nk == nk && m_nl == nl ;
}

bool NPYBase::hasItemShape(unsigned int nj, unsigned int nk, unsigned int nl)
{
    return m_nj == nj && m_nk == nk && m_nl == nl ;
}

bool NPYBase::hasItemSpec(NPYSpec* item_spec)
{
    return m_item_spec->isEqualTo(item_spec); 
}

bool NPYBase::hasShapeSpec(NPYSpec* shape_spec)
{
    return m_shape_spec->isEqualTo(shape_spec); 
}





void NPYBase::setNumItems(unsigned int ni)
{
    unsigned int orig = m_shape[0] ;
    assert(ni >= orig);

    LOG(debug) << "NPYBase::setNumItems"
              << " increase from " << orig << " to " << ni 
              ; 
 
    m_shape[0] = ni ; 
    m_ni = ni ; 
}


void NPYBase::reshape(int ni_, unsigned int nj, unsigned int nk, unsigned int nl)
{
    unsigned int nvals = std::max(1u,m_ni)*std::max(1u,m_nj)*std::max(1u,m_nk)*std::max(1u,m_nl) ; 
    unsigned int njkl  = std::max(1u,nj)*std::max(1u,nk)*std::max(1u,nl) ;
    unsigned int ni    = ni_ < 0 ? nvals/njkl : ni_ ;    // auto resizing of 1st dimension, when -ve

    unsigned int nvals2 = std::max(1u,ni)*std::max(1u,nj)*std::max(1u,nk)*std::max(1u,nl) ; 

    if(nvals != nvals2) LOG(fatal) << "NPYBase::reshape INVALID AS CHANGES COUNTS " 
                              << " nvals " << nvals
                              << " nvals2 " << nvals2
                              ;

    assert(nvals == nvals2 && "NPYBase::reshape cannot change number of values, just their addressing");

    LOG(debug) << "NPYBase::reshape (0 means no-dimension) "
              << "(" << m_ni << "," << m_nj << "," << m_nk << "," << m_nl << ")"
              << " --> "
              << "(" <<   ni << "," <<   nj << "," <<   nk << "," <<   nl << ")"
              ;

    m_shape.clear();
    if(ni > 0) m_shape.push_back(ni);
    if(nj > 0) m_shape.push_back(nj);
    if(nk > 0) m_shape.push_back(nk);
    if(nl > 0) m_shape.push_back(nl);

    updateDimensions();
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

    //ss << " ni " << m_ni ;
    //ss << " nj " << m_nj ;
    //ss << " nk " << m_nk ;
    //ss << " nl " << m_nl ;

    ss << " NumBytes(0) " << getNumBytes(0) ;
    ss << " NumBytes(1) " << getNumBytes(1) ;
    ss << " NumValues(0) " << getNumValues(0) ;
    ss << " NumValues(1) " << getNumValues(1) ;

    ss << m_metadata  ;

    return ss.str();
}

std::string NPYBase::path(const char* pfx, const char* gen, const char* tag, const char* det)
{
    std::stringstream ss ;
    ss << pfx << gen ;
    return path(ss.str().c_str(), tag, det );
}

std::string NPYBase::directory(const char* tfmt, const char* targ, const char* det)
{
    char typ[64];
    if(strchr (tfmt, '%' ) == NULL)
    {
        snprintf(typ, 64, "%s%s", tfmt, targ ); 
    }
    else
    { 
        snprintf(typ, 64, tfmt, targ ); 
    }
    std::string dir = directory(typ, det);
    return dir ; 
}





std::string NPYBase::directory(const char* typ, const char* det)
{
    std::string deftmpl(DEFAULT_DIR_TEMPLATE) ; 
    boost::replace_first(deftmpl, "$1", det );
    boost::replace_first(deftmpl, "$2", typ );
    std::string dir = fsutil::FormPath( deftmpl.c_str() ); 
    return dir ;
}
  
std::string NPYBase::path(const char* typ, const char* tag, const char* det)
{
/*
:param typ: object type name, eg oxcerenkov rxcerenkov 
:param tag: event tag, usually numerical 
:param det: detector tag, eg dyb, juno
*/

    std::string dir = directory(typ, det);
    dir += "/%s.npy" ; 

    char* tmpl = (char*)dir.c_str();
    char path_[256];
    snprintf(path_, 256, tmpl, tag );

    LOG(debug) << "NPYBase::path"
              << " typ " << typ
              << " tag " << tag
              << " det " << det
              << " DEFAULT_DIR_TEMPLATE " << DEFAULT_DIR_TEMPLATE
              << " tmpl " << tmpl
              << " path_ " << path_
              ;

    return path_ ;   
}

std::string NPYBase::path(const char* dir, const char* name)
{
    char path[256];
    snprintf(path, 256, "%s/%s", dir, name);

    //std::string path = fsutil::FormPath(dir, name);  
    // provides native style path with auto-prefixing based on envvar  
    return path ; 
}


