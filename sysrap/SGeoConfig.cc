#include <sstream>
#include <bitset>

#include "SBit.hh"
#include "SGeoConfig.hh"

unsigned long long SGeoConfig::EMM = SBit::FromEString("EMM", "~0");  

std::string SGeoConfig::Desc()
{
    std::stringstream ss ; 
    ss << " EMM " << SBit::HexString(EMM) << " 0x" << std::hex << EMM << std::dec ;  
    std::string s = ss.str(); 
    return s ; 
}

bool SGeoConfig::IsEnabledMergedMesh(unsigned mm) // static
{
    bool emptylistdefault = true ;   
    bool emm = true ;   
    if(mm < 64) 
    {   
        std::bitset<64> bs(EMM); 
        emm = bs.count() == 0 ? emptylistdefault : bs[mm] ;   
    }   
    return emm ; 
}

std::string SGeoConfig::DescEMM()
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < 64 ; i++) 
    {
        bool emm = SGeoConfig::IsEnabledMergedMesh(i) ; 
        if(emm) ss << i << " " ; 
    }
    std::string s = ss.str(); 
    return s ; 
}


