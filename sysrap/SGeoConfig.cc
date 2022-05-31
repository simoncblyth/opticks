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
#include "SName.h"

#include "PLOG.hh"

const plog::Severity SGeoConfig::LEVEL = PLOG::EnvLevel("SGeoConfig", "DEBUG"); 


unsigned long long SGeoConfig::_EMM = SBit::FromEString(kEMM, "~0");  
const char* SGeoConfig::_ELVSelection   = SSys::getenvvar(kELVSelection, nullptr ); 
const char* SGeoConfig::_SolidSelection = SSys::getenvvar(kSolidSelection, nullptr ); 
const char* SGeoConfig::_FlightConfig   = SSys::getenvvar(kFlightConfig  , nullptr ); 
const char* SGeoConfig::_ArglistPath    = SSys::getenvvar(kArglistPath  , nullptr ); 
const char* SGeoConfig::_CXSkipLV       = SSys::getenvvar(kCXSkipLV  , nullptr ); 

void SGeoConfig::SetELVSelection(  const char* es){  _ELVSelection   = es ? strdup(es) : nullptr ; }
void SGeoConfig::SetSolidSelection(const char* ss){  _SolidSelection = ss ? strdup(ss) : nullptr ; }
void SGeoConfig::SetFlightConfig(  const char* fc){  _FlightConfig   = fc ? strdup(fc) : nullptr ; }
void SGeoConfig::SetArglistPath(   const char* ap){  _ArglistPath    = ap ? strdup(ap) : nullptr ; }
void SGeoConfig::SetCXSkipLV(      const char* cx){  _CXSkipLV       = cx ? strdup(cx) : nullptr ; }

unsigned long long SGeoConfig::EnabledMergedMesh(){  return _EMM ; } 
const char* SGeoConfig::ELVSelection(){   return _ELVSelection ; }
const char* SGeoConfig::SolidSelection(){ return _SolidSelection ; }
const char* SGeoConfig::FlightConfig(){   return _FlightConfig ; }
const char* SGeoConfig::ArglistPath(){    return _ArglistPath ; }
const char* SGeoConfig::CXSkipLV(){       return _CXSkipLV ? _CXSkipLV : "" ; }


std::string SGeoConfig::Desc()
{
    std::stringstream ss ; 
    ss << std::endl ; 
    ss << std::setw(25) << kEMM << " : " << SBit::HexString(_EMM) << " 0x" << std::hex << _EMM << std::dec << std::endl ;
    ss << std::setw(25) << kELVSelection << " : " << ( _ELVSelection ? _ELVSelection : "-" ) << std::endl ;    
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

/**
SGeoConfig::GeometrySpecificSetup
-----------------------------------

This is invoked from the argumentless CSGFoundry::Load 
it detects if a geometry appears to be JUNO by the presence
of certain mesh names within it and if JUNO is detected
some JUNO specific static method calls are made.  

This avoids repeating these settings in tests or fiddling 
with envvars to configure these things. 

Previously did something simular using metadata in geocache
or from the Opticks setup code within detector specific code. 
However do not want to require writing cache and prefer to minimize 
detector specific Opticks  setup code as it is much easier 
to test in isolation than as an appendage to a detector framework. 

**/
void SGeoConfig::GeometrySpecificSetup(const SName* id)  // static
{
    const char* JUNO_names = "HamamatsuR12860sMask0x,HamamatsuR12860_PMT_20inch,NNVTMCPPMT_PMT_20inch" ;  
    bool JUNO_detect = id->hasNames(JUNO_names); 
    LOG(info) << " JUNO_detect " << JUNO_detect ; 
    if(JUNO_detect)
    {
        SetELVSelection("NNVTMCPPMTsMask_virtual0x,HamamatsuR12860sMask_virtual0x,mask_PMT_20inch_vetosMask_virtual0x");
    }
}


