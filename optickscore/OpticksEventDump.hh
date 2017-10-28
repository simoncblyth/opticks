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
       void Summary(const char* msg="OpticksEventDump::Summary");
       void dump(const char* msg="OpticksEventDump::dump");
       void dumpRecords(const char* msg="OpticksEventDump::dumpRecords");
       void dumpRecords(const char* msg, unsigned photon_id );

       void         dumpPhotonData(const char* msg="OpticksEventDump::dumpPhotonData");
       static void  dumpPhotonData(NPY<float>* photon_data);

   private:
       Opticks*          m_ok ; 
       OpticksEvent*     m_evt ; 
       OpticksEventStat* m_stat ; 
       bool              m_noload ; 
       RecordsNPY*       m_records ; 
};

#include "OKCORE_TAIL.hh"

 
