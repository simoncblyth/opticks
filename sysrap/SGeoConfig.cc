#include <iostream>
#include <iomanip>
#include <sstream>
#include <bitset>
#include <algorithm>
#include <cstring>

#include "SSys.hh"
#include "SStr.hh"
#include "SBit.hh"
#include "SGeoConfig.hh"

unsigned long long SGeoConfig::_EMM = SBit::FromEString(kEMM, "~0");  
const char* SGeoConfig::_SolidSelection = SSys::getenvvar(kSolidSelection, nullptr ); 
const char* SGeoConfig::_FlightConfig   = SSys::getenvvar(kFlightConfig  , nullptr ); 
const char* SGeoConfig::_ArglistPath    = SSys::getenvvar(kArglistPath  , nullptr ); 
const char* SGeoConfig::_CXSkipLV       = SSys::getenvvar(kCXSkipLV  , nullptr ); 

void SGeoConfig::SetSolidSelection(const char* ss){  _SolidSelection = ss ? strdup(ss) : nullptr ; }
void SGeoConfig::SetFlightConfig(  const char* fc){  _FlightConfig   = fc ? strdup(fc) : nullptr ; }
void SGeoConfig::SetArglistPath(   const char* ap){  _ArglistPath    = ap ? strdup(ap) : nullptr ; }
void SGeoConfig::SetCXSkipLV(      const char* cx){  _CXSkipLV       = cx ? strdup(cx) : nullptr ; }

unsigned long long SGeoConfig::EnabledMergedMesh(){  return _EMM ; } 
const char* SGeoConfig::SolidSelection(){ return _SolidSelection ; }
const char* SGeoConfig::FlightConfig(){   return _FlightConfig ; }
const char* SGeoConfig::ArglistPath(){    return _ArglistPath ; }
const char* SGeoConfig::CXSkipLV(){       return _CXSkipLV ? _CXSkipLV : "" ; }


std::string SGeoConfig::Desc()
{
    std::stringstream ss ; 
    ss << std::endl ; 
    ss << std::setw(25) << kEMM << " : " << SBit::HexString(_EMM) << " 0x" << std::hex << _EMM << std::dec << std::endl ;
    ss << std::setw(25) << kSolidSelection << " : " << ( _SolidSelection ? _SolidSelection : "-" ) << std::endl ;    
    ss << std::setw(25) << kFlightConfig << " : " << ( _FlightConfig ? _FlightConfig : "-" ) << std::endl ;    
    ss << std::setw(25) << kArglistPath << " : " << ( _ArglistPath ? _ArglistPath : "-" ) << std::endl ;    
    ss << std::setw(25) << kCXSkipLV << " : " << ( _CXSkipLV ? _CXSkipLV : "-" ) << std::endl ;    
    std::string s = ss.str(); 
    return s ; 
}

bool SGeoConfig::IsEnabledMergedMesh(unsigned mm) // static
{
    bool emptylistdefault = true ;   
    bool emm = true ;   
    if(mm < 64) 
    {   
        std::bitset<64> bs(_EMM); 
        emm = bs.count() == 0 ? emptylistdefault : bs[mm] ;   
    }   
    return emm ; 
}

bool SGeoConfig::IsCXSkipLV(int lv) // static
{
    if( _CXSkipLV == nullptr ) return false ; 
    std::vector<int> cxskip ;
    SStr::ISplit(_CXSkipLV, cxskip, ','); 
    return std::count( cxskip.begin(), cxskip.end(), lv ) == 1 ;   
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


std::vector<std::string>*  SGeoConfig::Arglist() 
{
    return SStr::LoadList( _ArglistPath, '\n' );  
}



