#include <sstream>
#include "NPYConfig.hpp"

std::string NPYConfig::OptionalExternals()
{
   std::stringstream ss ; 
#ifdef WITH_YoctoGL
    ss << "YoctoGL " ;
#endif
#ifdef WITH_DualContouringSample
    ss << "DualContouringSample " ;
#endif
#ifdef WITH_ImplicitMesher
    ss << "ImplicitMesher " ;
#endif
#ifdef WITH_CSGBSP
    ss << "CSGBSP " ;
#endif
    return ss.str();
}


