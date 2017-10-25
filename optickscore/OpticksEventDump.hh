#pragma once


template <typename T> class NPY ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OpticksEvent ; 
class OKCORE_API OpticksEventDump 
{
   public:
       OpticksEventDump( OpticksEvent* evt );
   public:
       void Summary(const char* msg="OpticksEventDump::Summary");
       void dump(const char* msg="OpticksEventDump::dump");
       void dumpRecords(const char* msg="OpticksEventDump::dumpRecords");

       void         dumpPhotonData(const char* msg="OpticksEventDump::dumpPhotonData");
       static void  dumpPhotonData(NPY<float>* photon_data);

   private:
       OpticksEvent* m_evt ; 
       bool          m_noload ; 
};

#include "OKCORE_TAIL.hh"

 
