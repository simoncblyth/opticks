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

/**

TEST=OpticksTwoTest om-t

while true; do OpticksTwoTest ; done
while OpticksTwoTest ; do echo -n ; done 

**/

#include "OPTICKS_LOG.hh"

#include "Opticks.hh"
#include "OpticksQuery.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
 
    Opticks ok(argc, argv);
    ok.configure();
    OpticksQuery* q = ok.getQuery();

    LOG(info)  << "idpath " << ok.getIdPath() ;
    LOG(info) << "q\n" <<  q->desc() ; 

    const char* key = "CX4GDMLTest.X4PhysicalVolume.World0xc15cfc0_PV.27c39be4e46a36ea28a3c4da52522c9e" ; 
    Opticks::SetKey(key);


    //Opticks ok1(0,0);
    Opticks ok1(argc,argv);
    ok1.configure();
    OpticksQuery* q1 = ok1.getQuery();

    LOG(info)  << "idpath1 " << ok1.getIdPath() ;
    LOG(info) << "q1\n" <<  ( q1 ? q1->desc() : "NULL-query" ) ; 


    return 0 ;   
}
