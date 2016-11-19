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
       bool isDbgPhoton(int record_id);
       bool isOtherPhoton(int record_id);
    public:
       const std::vector<int>&  getDbgIndex();
       const std::vector<int>&  getOtherIndex();
       std::string description();
   private:
       void postconfigure();
   private:
       Opticks*             m_ok ; 
       OpticksCfg<Opticks>* m_cfg ; 
       std::vector<int> m_debug_photon ; 
       std::vector<int> m_other_photon ; 

};

#include "OKCORE_HEAD.hh"


