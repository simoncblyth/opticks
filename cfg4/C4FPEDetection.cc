#include "C4FPEDetection.hh"
#include <iostream>


#if (defined(__GNUC__) && !defined(__clang__))
#ifdef __linux__
  //#include <features.h>
  #include <fenv.h>
  //#include <csignal>

void C4FPEDetection::InvalidOperationDetection_Disable()
{
      std::cout 
              << std::endl
              << "        "
              << "C4FPEDetection::InvalidOperationDetection_Disable"
              << std::endl
              << "        "
              << "############################################" << std::endl
              << "        "
              << "!!! WARNING - FPE detection is DISABLED  !!!" << std::endl
              << "        "
              << "############################################" << std::endl
              << std::endl
              ; 

    (void) fedisableexcept( FE_DIVBYZERO );
    (void) fedisableexcept( FE_INVALID );

}

#endif

#else

void C4FPEDetection::InvalidOperationDetection_Disable()
{
      std::cout 
              << std::endl
              << "  C4FPEDetection::InvalidOperationDetection_Disable      "
              << " NOT IMPLEMENTED "
              << std::endl
              ;
}

#endif


