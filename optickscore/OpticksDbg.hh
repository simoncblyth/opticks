#pragma once

#include <vector>
#include <string>

class Opticks ; 
template <typename T> class NPY ;
template <typename T> class OpticksCfg ;

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksDbg
============


The list of mask indices is used with aligned bi-simulation, to 
allow rerunning of single photons.


**/

class OKCORE_API OpticksDbg
{
       friend class Opticks ;  
    public:
       OpticksDbg(Opticks* ok);

       unsigned getNumDbgPhoton() const ;
       unsigned getNumOtherPhoton() const ;
       unsigned getNumMaskPhoton() const ;
       NPY<unsigned>* getMaskBuffer() const ;
       const std::vector<unsigned>&  getMask();
       unsigned getMaskIndex(unsigned idx) const ;

       bool isDbgPhoton(unsigned record_id);
       bool isOtherPhoton(unsigned record_id);
       bool isMaskPhoton(unsigned record_id);
    public:
       void loadNPY1(std::vector<unsigned>& vec, const char* path );
       const std::vector<unsigned>&  getDbgIndex();
       const std::vector<unsigned>&  getOtherIndex();
       std::string description();
   private:
       void postconfigure();
       void postconfigure(const std::string& spec, std::vector<unsigned>& ls);
   private:
       Opticks*              m_ok ; 
       OpticksCfg<Opticks>*  m_cfg ; 
       NPY<unsigned>*        m_mask_buffer ; 
       std::vector<unsigned> m_debug_photon ; 
       std::vector<unsigned> m_other_photon ; 
       std::vector<unsigned> m_mask ; 

};

#include "OKCORE_HEAD.hh"


