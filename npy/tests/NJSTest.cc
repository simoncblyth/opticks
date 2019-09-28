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
#include <map>

#include "NJS.hpp"
#include "OPTICKS_LOG.hh"


using json = nlohmann::json;


void test_write_read()
{
    const char* path = "$TMP/npy/NJSTest/test_write_read.json" ;

    std::map<std::string, int> m { {"one", 1}, {"two", 2}, {"three", 3} };
    json js(m) ; 
    NJS njs(js) ; 
    njs.write(path);

    NJS njs2 ; 
    njs2.read(path);
    njs2.dump();

    json& js2 = njs2.js() ;
    LOG(info) << "js2:" << js2.dump(4) ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_write_read();



    return 0 ; 
}

