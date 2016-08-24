#pragma once

class Opticks ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksIndexer {
   public:
       OpticksIndexer(Opticks* opticks);

   private:
       Opticks* m_opticks ; 


};

#include "OKCORE_TAIL.hh"

