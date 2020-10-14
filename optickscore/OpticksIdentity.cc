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

/**


::

    a = np.load("all_volume_identity.npy")
    tid = a[:,1]
    ridx = tid >> 24   
    pidx = np.where( ridx == 0,                       0, ( tid >>  8 ) & 0xffff ) 
    oidx = np.where( ridx == 0, ( tid >> 0 ) & 0xffffff, ( tid >> 0  ) & 0xff   )

**/


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


unsigned OpticksIdentity::RepeatIndex(unsigned identifier) // static 
{
    return ( identifier >> 24 ) & 0xff ; 
}
unsigned OpticksIdentity::PlacementIndex(unsigned identifier) // static 
{
    unsigned ridx = RepeatIndex(identifier); 
    return ridx == 0 ? 0 : (  identifier >>  8 ) & 0xffff ; 
}
unsigned OpticksIdentity::OffsetIndex(unsigned identifier) // static 
{
    unsigned ridx = RepeatIndex(identifier); 
    return ridx == 0 ? ( identifier >> 0 ) & 0xffffff : ( identifier >>  0 ) & 0xff   ; 
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


std::string OpticksIdentity::Desc(unsigned identifier) // static 
{
    unsigned repeat_index ; 
    unsigned placement_index ;
    unsigned offset_index ;

    Decode(identifier, repeat_index, placement_index, offset_index); 

    std::stringstream ss ; 
    ss << "rpo(" 
       << repeat_index 
       << " "
       << placement_index 
       << " "
       << offset_index 
       << ")"
       << " " << std::setw(7) << std::hex << identifier
       ;
    return ss.str(); 
}



std::string OpticksIdentity::Desc(const char* label, const glm::uvec4& id ) // static
{
    std::stringstream ss ; 
    ss  
        << label
        << "[" 
        << glm::to_string(id)
        << ";"
        << Desc(id.y)
        << "]" 
        ;   
    return ss.str(); 
}

std::string OpticksIdentity::desc() const 
{
    std::stringstream ss ; 
    ss << "OpticksIdentity(" 
       << m_repeat_index 
       << " "
       << m_placement_index 
       << " "
       << m_offset_index 
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
 


