#include <sstream>
#include "CTrk.hh"

unsigned   CTrk::Photon_id( unsigned packed) { return ( packed & 0x3fffffff ) ; }    // static 
char       CTrk::Gentype(   unsigned packed) { return ( packed & (0x1 << 31) ) ? 'C' : 'S' ;  } // static
bool       CTrk::Reemission(unsigned packed) { return ( packed & (0x1 << 30) ) ;  }   // static 


CTrk::CTrk( const CTrk& other)
    :
    m_packed( other.packed() )
{
}

CTrk::CTrk( unsigned packed )
    :
    m_packed( packed )
{
}

CTrk::CTrk( unsigned photon_id_ , char gentype_, bool reemission_ )
    :
    m_packed((photon_id_ & 0x3fffffff) | unsigned(gentype_ == 'C') << 31 | unsigned(reemission_) << 30 )   
{
}

unsigned   CTrk::packed()     const { return m_packed ; }
unsigned   CTrk::photon_id() const { return Photon_id(m_packed) ; }
char       CTrk::gentype()   const { return Gentype(m_packed) ; }
bool       CTrk::reemission() const { return Reemission(m_packed) ; }

std::string CTrk::desc() const 
{ 
    std::stringstream ss ; 
    ss << "CTrackInfo"
       << " gentype " << gentype()
       << " photon_id " << photon_id()
       << " reemission " << reemission() 
       << " packed " << packed()
       ;  
    std::string s = ss.str(); 
    return s ; 
}


