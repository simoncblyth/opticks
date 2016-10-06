#pragma once

#include <vector>
#include <string>

class Opticks ; 
template <typename T> class OpticksCfg ;

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksDbg
============

**/

class OKCORE_API OpticksDbg
{
       friend class Opticks ;  
    public:
       OpticksDbg(Opticks* ok);
       bool isDbgPhoton(int photon_id);
       const std::vector<int>&  getDbgIndex();
       std::string description();
   private:
       void postconfigure();
   private:
       Opticks*             m_ok ; 
       OpticksCfg<Opticks>* m_cfg ; 
       std::vector<int> m_debug_photon ; 

};

#include "OKCORE_HEAD.hh"


