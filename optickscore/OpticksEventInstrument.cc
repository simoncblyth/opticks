/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <iostream>

#include "NPY.hpp"
#include "RecordsNPY.hpp"
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"

#include "OpticksEventInstrument.hh"

#include "PLOG.hh"



RecordsNPY* OpticksEventInstrument::CreateRecordsNPY(const OpticksEvent* evt) // static
{
    LOG(verbose) << "OpticksEventInstrument::CreateRecordsNPY start" ; 

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

    LOG(verbose) << "OpticksEventInstrument::CreateRecordsNPY done" ; 

    return rec ; 
} 



