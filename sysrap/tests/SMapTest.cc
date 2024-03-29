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


#include <string>
#include <csignal>
#include "SMap.hh"

#include "SYSRAP_LOG.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    typedef std::string K ; 
    typedef unsigned long long V ; 

    std::map<K,V> m ; 

    V v =  0xdeadbeefdeadbeefull ; 

    m["hello"] = v ; 
    m["world"] = v ; 
    m["other"] = 0x4full ; 
    m["yo"] = 0xffffull ; 


    unsigned nv = SMap<K,V>::ValueCount(m, v) ; 

    bool nv_expect = nv == 2 ;
    assert( nv_expect );
    if(!nv_expect) std::raise(SIGINT); 

    bool dump = true ; 

    std::vector<K> keys ; 
    SMap<K,V>::FindKeys(m, keys, v, dump ) ; 

    assert( keys.size() == 2 ) ;


    return 0 ;
}
