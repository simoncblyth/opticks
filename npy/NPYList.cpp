#include "BFile.hh"
#include "BStr.hh"

#include "NPYBase.hpp"
#include "NPYSpec.hpp"
#include "NPYSpecList.hpp"
#include "NPYList.hpp"

#include "PLOG.hh"


NPYList::NPYList(const NPYSpecList* specs)
    :
    m_specs(specs)
{
    init();
}

void NPYList::init()
{
    m_buf.fill(NULL); 
}

const NPYSpec* NPYList::getBufferSpec( int bid) const
{
    return m_specs->getByIdx( (unsigned)bid );  
}
NPYBase::Type_t NPYList::getBufferType(int bid) const
{
    return getBufferSpec(bid)->getType() ; 
}
const char* NPYList::getBufferName(int bid) const
{
    return getBufferSpec(bid)->getName() ; 
}

std::string NPYList::getBufferPath( const char* treedir, int bid ) const
{
    std::string path = BFile::FormPath(treedir, getBufferName(bid) ) ;
    return path ; 
}

void NPYList::saveBuffer(const char* treedir, int bid ) const
{
    const NPYSpec* spec = getBufferSpec(bid) ; 
    bool optional = spec->isOptional() ; 
    NPYBase* buffer = getBuffer(bid); 
    if( !optional && buffer == NULL )
    {
        LOG(fatal) << " non-optional buffer is NULL  " << getBufferName(bid) ; 
        assert(0) ; 
    }
    if( buffer == NULL ) return ; 
    std::string path = getBufferPath(treedir, bid); 
    buffer->save(path.c_str());  
}

void NPYList::loadBuffer(const char* treedir, int bid )
{
    const NPYSpec* spec = getBufferSpec(bid) ; 
    bool optional = spec->isOptional() ; 

    std::string path = getBufferPath(treedir, bid); 

    bool exists = BFile::ExistsFile(path.c_str()) ;
    if(!optional && !exists)
    {
        LOG(fatal) << " non-optional buffer does not exist " << path ;
        assert(0) ; 
    }
    if(!exists) return ; 

    NPYBase::Type_t type = getBufferType(bid); 
    NPYBase* buffer = NPYBase::Load( path.c_str(), type );     
    assert( buffer->hasItemSpec( spec )); 

    setBuffer(bid, buffer ); 
    NPYBase* buffer2 = getBuffer(bid) ; 
    assert( buffer == buffer2 ); 
}   

void NPYList::setBuffer( int bid, NPYBase* buffer )
{
    NPYBase* prior = getBuffer(bid) ; 
    if(prior) LOG(warning) << "replacing prior buffer" ; 
    assert( bid > -1 && bid < MAX_BUF ) ; 
    m_buf[bid] = buffer ; 
}

NPYBase* NPYList::getBuffer(int bid) const 
{
    NPYBase* buffer = m_buf[bid] ; 
    return buffer ; 
}

std::string NPYList::getBufferShape(int bid) const 
{
    NPYBase* buffer = getBuffer(bid);
    return buffer ? buffer->getShapeString() : "-" ; 
}

unsigned NPYList::getNumItems(int bid) const 
{
    const NPYSpec* spec = getBufferSpec(bid) ; 
    bool optional = spec->isOptional() ; 
    NPYBase* buffer = getBuffer(bid);

    if( buffer == NULL && !optional )
    {
        LOG(fatal) << "non-optional buffer does not exist " ; 
        assert(0); 
    }
    return buffer ? buffer->getNumItems() : 0 ; 
}


void NPYList::initBuffer(int bid, int ni, bool zero) 
{
    const NPYSpec* spec = getBufferSpec(bid) ; 
    NPYBase* buffer = NPYBase::Make( ni, spec, zero );  
    setBuffer( bid, buffer );  
}


std::string NPYList::desc() const 
{
    std::stringstream ss ; 
    ss << "NPYList" ; 
    for(int i=0 ; i < MAX_BUF ; i++)
    {
        NPYBase* buf = getBuffer(i); 
        if(buf==NULL) continue ; 
        ss 
           << " "
           << getBufferName(i)
           << " "
           << buf->getShapeString()
           << "  " 
           ;
    } 
    return ss.str() ;
}

