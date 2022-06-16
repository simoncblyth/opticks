#pragma once

struct sdomain
{
    static constexpr const float    DOMAIN_LOW  = 60.f ; 
    static constexpr const float    DOMAIN_HIGH = 820.f ; 
    static constexpr const float    DOMAIN_STEP = 20.f ; 
    static constexpr const unsigned DOMAIN_LENGTH = 39 ; 
    static constexpr const char     DOMAIN_TYPE = 'F' ;   // 'C'
    static constexpr const float    FINE_DOMAIN_STEP = 1.f ; 
    static constexpr const unsigned FINE_DOMAIN_LENGTH = 761 ; 
    static constexpr unsigned DomainLength(){  return DOMAIN_TYPE == 'F' ? FINE_DOMAIN_LENGTH : DOMAIN_LENGTH ; }
};



