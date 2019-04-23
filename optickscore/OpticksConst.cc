#include <sstream>

#include "OpticksConst.hh"


const char* OpticksConst::BNDIDX_NAME_  = "Boundary_Index" ;
const char* OpticksConst::SEQHIS_NAME_  = "History_Sequence" ;
const char* OpticksConst::SEQMAT_NAME_  = "Material_Sequence" ;


std::string OpticksConst::describeModifiers(unsigned int modifiers)
{
    std::stringstream ss ; 
    if(modifiers & e_shift)   ss << "shift " ; 
    if(modifiers & e_control) ss << "control " ; 
    if(modifiers & e_option)  ss << "option " ; 
    if(modifiers & e_command) ss << "command " ;
    return ss.str(); 
}
bool OpticksConst::isShift(unsigned int modifiers) { return 0 != (modifiers & e_shift) ; }
bool OpticksConst::isOption(unsigned int modifiers) { return 0 != (modifiers & e_option) ; }
bool OpticksConst::isShiftOption(unsigned int modifiers) { return isShift(modifiers) && isOption(modifiers) ; }
bool OpticksConst::isCommand(unsigned int modifiers) { return 0 != (modifiers & e_command) ; }
bool OpticksConst::isControl(unsigned int modifiers) { return 0 != (modifiers & e_control) ; }


const char OpticksConst::GEOCODE_ANALYTIC = 'A';
const char OpticksConst::GEOCODE_TRIANGULATED = 'T' ;
const char OpticksConst::GEOCODE_RTXTRIANGLES = 'R' ;
const char OpticksConst::GEOCODE_SKIP = 'K' ;


