#pragma once


template <typename T> class NPY ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class RecordsNPY ; 
class Opticks ;
class OpticksEvent ;
class OpticksEventStat ;
 
class OKCORE_API OpticksEventDump 
{
   public:
       OpticksEventDump( OpticksEvent* evt );
   private:
       void init();
   public:
       void Summary(const char* msg="OpticksEventDump::Summary") const ;
       void dump(unsigned photon_id) const ;
       unsigned getNumPhotons() const ;
   private:
       void dumpRecords(unsigned photon_id ) const ;
       void dumpPhotonData(unsigned photon_id) const ;
   private:
       Opticks*          m_ok ; 
       OpticksEvent*     m_evt ; 
       OpticksEventStat* m_stat ; 
       bool              m_noload ; 
       RecordsNPY*       m_records ; 
       NPY<float>*       m_photons ; 
       NPY<unsigned long long>* m_seq ;
       unsigned          m_num_photons ; 
};

#include "OKCORE_TAIL.hh"

 
