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

#include "OPTICKS_LOG.hh"

#include "No.hpp"
#include "NNodeCollector.hpp"
#include "NTreeAnalyse.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    no* g = new no {"g", NULL, NULL } ;
    no* f = new no {"f", NULL, NULL } ;
    no* e = new no {"e", NULL, NULL } ;
    no* d = new no {"d", NULL, NULL } ;
    no* c = new no {"c",  f,  g } ;
    no* b = new no {"b",  d,  e } ;
    no* a = new no {"a",  b,  c } ; 
   
    LOG(info) << a->desc() ; 

    NTreeAnalyse<no> ana(a); 
    ana.nodes->dump() ; 

    LOG(info) << ana.desc() ; 


    return 0 ; 
}


