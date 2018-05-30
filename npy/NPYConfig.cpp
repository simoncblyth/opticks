#include <sstream>
#include "NPYConfig.hpp"

std::string NPYConfig::OptionalExternals()
{
   std::stringstream ss ; 
#ifdef OPTICKS_YoctoGL
    ss << "YoctoGL " ;
#endif
#ifdef OPTICKS_DualContouringSample
    ss << "DualContouringSample " ;
#endif
#ifdef OPTICKS_ImplicitMesher
    ss << "ImplicitMesher " ;
#endif
#ifdef OPTICKS_CSGBSP
    ss << "CSGBSP " ;
#endif
    return ss.str();
}


