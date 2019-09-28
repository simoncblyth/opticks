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

#include "BTimesTable.hh"
#include "BFile.hh"

#include "OPTICKS_LOG.hh"


void load_dump()
{
    LOG(info) ; 
    const char* dir = "$TMP/tboolean-box/evt/tboolean-box/torch/2" ; 

    BTimesTable* tt = new BTimesTable("Time,DeltaTime"); 
    tt->load(dir);  // attempts to load t_absolute.ini and t_delta.ini from the directory : corresponding to table columns
    tt->dump();
}

void quad_add()
{
    LOG(info) ; 
    const char* columns = "A,B,C,D" ; 

    BTimesTable* tt = new BTimesTable(columns); 

    for(unsigned i=0 ; i < 20 ; i++) tt->add(i, i*0, i*10, i*20, i*30 );
    tt->add("hello", 42, 42, 42, 42 );

    const char* check = "check" ;
    tt->add(check, 43, 43, 43, 43 );

    tt->dump();
    const char* dir = "$TMP/boostrap/BTimesTableTest/quad_add" ;
    tt->save(dir) ; 

    BTimesTable* zz = new BTimesTable(columns); 
    zz->load(dir);
    zz->dump();
}

void filter_dump()
{
    LOG(info) ; 
    const char* columns = "A,B,C,D" ; 
    BTimesTable* tt = new BTimesTable(columns); 

    tt->add("red", 0, 10, 20, 30, 0 );
    tt->add("red", 0, 10, 20, 30, 1 );
    tt->add("red", 0, 10, 20, 30, 2 );

    tt->add("gred", 0, 10, 20, 30, 0 );
    tt->add("gred", 0, 10, 20, 30, 1 );
    tt->add("gred", 0, 10, 20, 30, 2 );
   
    tt->add("rouge", 0, 10, 20, 30, 0 );
    tt->add("rouge", 0, 10, 20, 30, 1 );
    tt->add("rouge", 0, 10, 20, 30, 2 );
     
    tt->add("rout", 0, 10, 20, 30, 0 );
    tt->add("rout", 0, 10, 20, 30, 1 );
    tt->add("rout", 0, 10, 20, 30, 2 );

    tt->add("Opticks::Opticks", 0, 10, 20, 30, 0 );
    tt->add("OPropagator::launch", 0, 10, 20, 30, 0 );
    tt->add("OPropagator::launch", 0, 10, 20, 30, 1 );
    

    tt->dump("unfiltered");

    tt->dump("starting with ro", "ro");
 
    tt->dump("starting with OPropagator::launch", "OPropagator::launch");
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    load_dump();
    quad_add();
    filter_dump();

    return 0 ; 
}
