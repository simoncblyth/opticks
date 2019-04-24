#pragma once

#include <string>
#include "OKCORE_API_EXPORT.hh"

class OKCORE_API OpticksConst {
   public:
       static const char* BNDIDX_NAME_ ;
       static const char* SEQHIS_NAME_ ;
       static const char* SEQMAT_NAME_ ;
   public:
       enum { 
              e_shift   = 1 << 0,  
              e_control = 1 << 1,  
              e_option  = 1 << 2,  
              e_command = 1 << 3 
            } ; 
      // GLFW_MOD_SHIFT
      // GLFW_MOD_CONTROL
      // GLFW_MOD_ALT
      // GLFW_MOD_SUPER

       static bool isShift(unsigned int modifiers);
       static bool isControl(unsigned int modifiers);
       static bool isShiftOption(unsigned int modifiers);
       static bool isCommand(unsigned int modifiers);  // <-- Linux super key often redefined, unlike macOS command key so not so useful
       static bool isOption(unsigned int modifiers);
       static std::string describeModifiers(unsigned int modifiers);
   public:
       static const char GEOCODE_ANALYTIC ;
       static const char GEOCODE_TRIANGULATED ;
       static const char GEOCODE_GEOMETRYTRIANGLES ;
       static const char GEOCODE_SKIP  ;

};




 
