#pragma once

class Opticks ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

//
// skeleton to hold high level Opticks instances
// intended to take over from the organic ggv-/App
// in a more modular way with helper classes for 
// such things as index presentation prep
//

class OKCORE_API OpticksApp {
   public:
       OpticksApp(Opticks* opticks);
   private:
       Opticks* m_opticks ; 
};

#include "OKCORE_TAIL.hh"



