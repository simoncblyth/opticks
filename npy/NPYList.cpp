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
    m_locked.fill(false); 
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

void NPYList::saveBuffer(const char* treedir, int bid, const char* msg ) const
{
    const NPYSpec* spec = getBufferSpec(bid) ; 
    bool optional = spec->isOptional() ; 
    int verbosity = spec->getVerbosity() ; 
    NPYBase* buffer = getBuffer(bid); 
    if( !optional && buffer == NULL )
    {
        LOG(fatal) << " non-optional buffer is NULL  " << getBufferName(bid) ; 
        assert(0) ; 
    }
    if( buffer == NULL ) return ; 
    unsigned ni = buffer->getNumItems(); 
    std::string path = getBufferPath(treedir, bid); 

    if(verbosity > 2)
    LOG(info) 
           << " spec.verbosity " << verbosity 
           <<  " save " << path 
           <<  " numItems " << ni 
           << " msg " << ( msg ? msg : "-" )
           ; 

    if( optional && ni == 0) return ; 
    buffer->save(path.c_str());  
}

void NPYList::loadBuffer(const char* treedir, int bid, const char* msg)
{
    const NPYSpec* spec = getBufferSpec(bid) ; 
    if(isLocked(bid))
    {
        LOG(fatal) << "loadBuffer NOT ALLOWED :  BUFFER IS LOCKED " << spec->getName() ; 
        assert(0); 
    }  

    bool optional = spec->isOptional() ; 
    int verbosity = spec->getVerbosity() ; 

    std::string path = getBufferPath(treedir, bid); 
    bool exists = BFile::ExistsFile(path.c_str()) ;
    if(!optional && !exists)
    {
        LOG(fatal) << " non-optional buffer does not exist " << path ;
        assert(0) ; 
    }
    if(!exists) return ; 

    if(verbosity > 2 )
        LOG(info) 
             << " spec.verbosity " << verbosity 
             <<  " loaded " << path 
             << " msg " << ( msg ? msg : "-" )
             ; 


    NPYBase::Type_t type = getBufferType(bid); 
    NPYBase* buffer = NPYBase::Load( path.c_str(), type );     
    assert( buffer->hasItemSpec( spec )); 

    setBuffer(bid, buffer ); 
    NPYBase* buffer2 = getBuffer(bid) ; 
    assert( buffer == buffer2 ); 
}   

void NPYList::setBuffer( int bid, NPYBase* buffer, const char* msg )
{
    const NPYSpec* spec = getBufferSpec(bid) ; 
    if(isLocked(bid))
    {
        LOG(fatal) << "setBuffer NOT ALLOWED :  BUFFER IS LOCKED " << spec->getName() ; 
        assert(0); 
    }  

    NPYBase* prior = getBuffer(bid) ; 
    if(prior) 
    { 
        LOG(error) << "replacing " << spec->getName() << " buffer "   
                   << " prior " << prior->getShapeString()
                   << " buffer " << buffer->getShapeString()
                   << " msg " << ( msg ? msg : "-" )
                   ; 
        //assert(0);   NCSGList trips this with its re-export following an adjustToFit of the container
    } 
    assert( bid > -1 && bid < MAX_BUF ) ; 
    m_buf[bid] = buffer ; 
}

void NPYList::setLocked( int bid, bool locked )
{
    assert( bid > -1 && bid < MAX_BUF ) ; 
    m_locked[bid] = locked ; 
}

bool NPYList::isLocked( int bid )
{
    assert( bid > -1 && bid < MAX_BUF ) ; 
    return m_locked[bid] ; 
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


void NPYList::initBuffer(int bid, int ni, bool zero, const char* msg) 
{
    const NPYSpec* spec = getBufferSpec(bid) ; 
    if(isLocked(bid))
    {
        LOG(fatal) << "initBuffer NOT ALLOWED :  BUFFER IS LOCKED " << spec->getName() ; 
        assert(0); 
    }  

    int verbosity = spec->getVerbosity() ; 
    if( verbosity > 2 ) 
         LOG(info) << " initBuffer " 
                   << " spec.verbosity " << verbosity 
                   << " msg " << ( msg ? msg : "NULL" ) 
                   << " name " << spec->getName() 
                  ; 


    NPYBase* buffer = NPYBase::Make( ni, spec, zero );  
    setBuffer( bid, buffer, msg );  
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

