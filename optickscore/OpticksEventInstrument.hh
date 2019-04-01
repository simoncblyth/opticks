#pragma once

class RecordsNPY ; 
class OpticksEvent ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksEventInstrument
========================

**/

class OKCORE_API OpticksEventInstrument
{
    public:
        static RecordsNPY* CreateRecordsNPY(const OpticksEvent* evt);
};


#include "OKCORE_TAIL.hh"


