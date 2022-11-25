// name=sbitmask ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>
#include <bitset>
#include <string>
#include <sstream>
#include <cstdint>

#include "sbitmask.h"

template<typename T>
std::string desc(int i, T mask)
{
    std::stringstream ss ; 
    ss << " i " << std::setw(3) << i  ; 

    if( sizeof(T) == 4 )
    {
         ss << " mask " << std::setw(10) << mask
            << " (0x) " << std::hex << std::setw(10) << mask << std::dec
            << " (0b) " << std::bitset<32>(mask).to_string()   
            ;
    }
    else if( sizeof(T) == 8 )
    {
        ss << " mask " << std::setw(25) << mask
           << " (0x) " << std::hex << std::setw(16) << mask << std::dec
           << " (0b) " << std::bitset<64>(mask).to_string()   
           ;
    }
    std::string s = ss.str(); 
    return s ; 
}

template<typename T>
std::string desc(bool dbg=false)
{
    int n = sizeof(T)*CHAR_BIT ; 
    std::stringstream ss ; 
    ss << "desc_" << n << std::endl ; 
    for(int i=0 ; i <= n ; i++)
    {
        T mask = sbitmask<T>(i) ; 
        T mask0 = sbitmask_0<T>(i) ;  // all bits set, except i=0 which is all bits unset 
        T mask1 = sbitmask_1<T>(i) ; 
        T mask2 = sbitmask_2<T>(i) ; 

        if(dbg)
        { 
            ss << "mk " << desc<T>(i, mask ) << std::endl ; 
            ss << "m0 " << desc<T>(i, mask0 ) << std::endl ; 
            ss << "m1 " << desc<T>(i, mask1 ) << std::endl ; 
            ss << "m2 " << desc<T>(i, mask2 ) << std::endl ; 
            ss << std::endl ;  
        }
        else
        {
            ss << "mk " << desc<T>(i, mask ) << std::endl ; 
        }
    }
    std::string s = ss.str(); 
    return s ; 
}

int main(int argc, char** argv)
{
    bool dbg=false ; 
    std::cout << desc<uint32_t>(dbg);   
    std::cout << desc<uint64_t>(dbg);   
    return 0 ; 
}
