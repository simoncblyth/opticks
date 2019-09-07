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

// TEST=OK_PROFILE_Test om-t

#include "Opticks.hh"
#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc,argv);

    Opticks* m_ok = new Opticks(argc, argv); 
    m_ok->configure(); 

    
    LOG(info) << argv[0] ;

    std::vector<double> times ; 
    OK_PROFILE("head");   


    for(unsigned i=0 ; i < 100 ; i++)
    {
        OK_PROFILE("body");   
    } 
    OK_PROFILE("tail");   


    m_ok->dumpProfile();

    //m_ok->saveProfile(); 


    return 0 ;
}


