#include <sstream>
#include <iomanip>

#include "SComp.h"
#include "SSys.hh"
#include "SFrameConfig.hh"

unsigned SFrameConfig::_FrameMask = SComp::Mask(SSys::getenvvar(kFrameMask, _FrameMaskDefault )) ;   

unsigned SFrameConfig::FrameMask(){  return _FrameMask; } 
void SFrameConfig::SetFrameMask(const char* names, char delim){  _FrameMask = SComp::Mask(names,delim) ; }

std::string SFrameConfig::FrameMaskLabel(){ return SComp::Desc( _FrameMask ) ; }

  
std::string SFrameConfig::Desc()
{
    std::stringstream ss ; 
    ss << "SFrameConfig::Desc" << std::endl 
       << std::setw(25) << kFrameMask
       << std::setw(20) << " FrameMask " << " : " << FrameMask() << std::endl 
       << std::setw(25) << ""
       << std::setw(20) << " FrameMaskLabel " << " : " << FrameMaskLabel() << std::endl 
       << std::setw(25) << ""
       << std::setw(20) << " _FrameMaskDefault " << " : " << _FrameMaskDefault << std::endl 
       << std::setw(25) << ""
       << std::setw(20) << " _FrameMaskAll " << " : " << _FrameMaskAll << std::endl 
       ;
    std::string s = ss.str(); 
    return s ; 
}



