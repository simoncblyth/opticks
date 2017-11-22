
#include <iostream>
#include "RecordsNPY.hpp"
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"

#include "OpticksEventInstrument.hh"

#include "PLOG.hh"



RecordsNPY* OpticksEventInstrument::CreateRecordsNPY(const OpticksEvent* evt) // static
{
    LOG(trace) << "OpticksEventInstrument::CreateRecordsNPY start" ; 

    if(!evt || evt->isNoLoad()) return NULL ; 

    Opticks* ok = evt->getOpticks();

    NPY<short>* rx = evt->getRecordData();
    assert(rx && rx->hasData());
    unsigned maxrec = evt->getMaxRec() ;

    Types* types = ok->getTypes();
    Typ* typ = ok->getTyp();


    RecordsNPY* rec = new RecordsNPY(rx, maxrec);

    rec->setTypes(types);
    rec->setTyp(typ);
    rec->setDomains(evt->getFDomain()) ;

    LOG(info) << "OpticksEventInstrument::CreateRecordsNPY " 
              << " shape " << rx->getShapeString() 
              ;

    LOG(trace) << "OpticksEventInstrument::CreateRecordsNPY done" ; 

    return rec ; 
} 



