#include <cassert>
#include <sstream>
#include "PLOG.hh"
#include "OpticksIdentity.hh"

const plog::Severity OpticksIdentity::LEVEL = PLOG::EnvLevel("OpticksIdentity", "DEBUG"); 

unsigned OpticksIdentity::Encode(unsigned repeat_index, unsigned placement_index, unsigned offset_index) // static 
{
    unsigned encoded_identifier = 0 ; 

    LOG(LEVEL) 
        << " repeat_index " << repeat_index 
        << " placement_index " << placement_index 
        << " offset_index " << offset_index 
        ;

    if( repeat_index > 0)
    {

        bool repeat_ok = (repeat_index    & 0xff)   == repeat_index ; 
        bool placement_ok = (placement_index & 0xffff) == placement_index  ;
        bool offset_ok = (offset_index    & 0xff)   == offset_index ; 
 
        if(!repeat_ok || !placement_ok || !offset_ok )
            LOG(fatal) 
                << " repeat_index " << repeat_index 
                << " placement_index " << placement_index 
                << " offset_index " << offset_index 
                ;

        assert( repeat_ok ); 
        assert( placement_ok ); 
        assert( offset_ok ); 

        encoded_identifier = 
              ( repeat_index     << 24 ) | 
              ( placement_index  << 8  ) |
              ( offset_index     << 0  ) 
              ; 
    }
    else if( repeat_index == 0)
    {
        bool repeat_ok = (repeat_index    & 0xff)   == repeat_index ; 
        bool placement_ok = placement_index == 0  ;
        bool offset_ok = (offset_index    & 0xffffff) == offset_index ;  

        if(!repeat_ok || !placement_ok || !offset_ok )
            LOG(fatal) 
                << " repeat_index " << repeat_index 
                << " placement_index " << placement_index 
                << " offset_index " << offset_index 
                ;

        assert( repeat_ok ); 
        assert( placement_ok ); 
        assert( offset_ok ); 

        encoded_identifier = 
              ( repeat_index << 24 ) | 
              ( offset_index << 0 ) 
              ; 
    }
    return encoded_identifier ; 
}

bool OpticksIdentity::Decode(unsigned identifier, unsigned& repeat_index, unsigned& placement_index, unsigned& offset_index ) // static 
{
    repeat_index    = ( identifier >> 24 ) & 0xff ; 
    if( repeat_index > 0 )  
    {
        placement_index = ( identifier >>  8 ) & 0xffff ; 
        offset_index    = ( identifier >>  0 ) & 0xff ; 
    }
    else if( repeat_index == 0 )
    {
        placement_index = 0 ; 
        offset_index = ( identifier >> 0 ) & 0xffffff ; 
    }
    return true ; 
}

OpticksIdentity::OpticksIdentity(unsigned repeat_index, unsigned placement_index, unsigned offset_index)
    :
    m_repeat_index(repeat_index),
    m_placement_index(placement_index),
    m_offset_index(offset_index),
    m_encoded_identifier(OpticksIdentity::Encode(repeat_index,placement_index,offset_index)),
    m_decoded(false) 
{
}

OpticksIdentity::OpticksIdentity(unsigned identifier)
    :
    m_repeat_index(0),
    m_placement_index(0),
    m_offset_index(0),
    m_encoded_identifier(identifier),
    m_decoded(OpticksIdentity::Decode(identifier, m_repeat_index, m_placement_index, m_offset_index))
{
} 

std::string OpticksIdentity::desc() const 
{
    std::stringstream ss ; 
    ss << "OpticksIdentity(" 
       << std::setw(2) << m_repeat_index 
       << ","
       << std::setw(6) << m_placement_index 
       << ","
       << std::setw(6) << m_offset_index 
       << ")"
       << " " << std::setw(10) << std::dec << m_encoded_identifier
       << " " << std::setw(10) << std::hex << m_encoded_identifier
       ;
    return ss.str(); 
}


unsigned OpticksIdentity::getRepeatIndex() const 
{
    return m_repeat_index ; 
}
unsigned OpticksIdentity::getPlacementIndex() const 
{
    return m_placement_index ; 
}
unsigned OpticksIdentity::getOffsetIndex() const 
{
    return m_offset_index ; 
}
unsigned OpticksIdentity::getEncodedIdentifier() const 
{
    return m_encoded_identifier ; 
} 
 


