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

Canonical m_dbg instance is ctor resident of Opticks.

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
       unsigned getNumX4PolySkip() const ;
       unsigned getNumCSGSkipLV() const ;
    public:
       NPY<unsigned>* getMaskBuffer() const ;
       const std::vector<unsigned>&  getMask();
       unsigned getMaskIndex(unsigned idx) const ;
    public:
       static bool IsListed(unsigned idx, const std::vector<unsigned>& ls, bool emptylistdefault);
       bool isDbgPhoton(unsigned record_id) const ;
       bool isOtherPhoton(unsigned record_id) const ;
       bool isMaskPhoton(unsigned record_id) const ;
       bool isX4PolySkip(unsigned lvIdx) const ;
       bool isCSGSkipLV(unsigned lvIdx) const ;   // --csgskiplv
       bool isEnabledMergedMesh(unsigned mm) const ;
    public:
       unsigned getInstanceModulo(unsigned mm) const ; 
    public:
       void loadNPY1(std::vector<unsigned>& vec, const char* path );
       const std::vector<unsigned>&  getDbgIndex();
       const std::vector<unsigned>&  getOtherIndex();
       std::string description();
   private:
       void postconfigure();
       void postconfigure(const std::string& spec, std::vector<unsigned>& ls);
       void postconfigure(const std::string& spec, std::vector<std::pair<int, int> >& pairs ) ;
   private:
       Opticks*              m_ok ; 
       OpticksCfg<Opticks>*  m_cfg ; 
       NPY<unsigned>*        m_mask_buffer ; 
       std::vector<unsigned> m_debug_photon ; 
       std::vector<unsigned> m_other_photon ; 
       std::vector<unsigned> m_mask ; 
       std::vector<unsigned> m_x4polyskip ; 
       std::vector<unsigned> m_csgskiplv ; 
       std::vector<unsigned> m_enabledmergedmesh ; 
       std::vector<std::pair<int,int> > m_instancemodulo ;   // (1,5),(2,10)  modulo scaledown for each mergedmesh index 

};

#include "OKCORE_HEAD.hh"


