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
#include <iostream>
#include <fstream>

#include "YGLTF.h"

using ygltf::json ; 


// https://github.com/nlohmann/json/

struct sphere
{
    std::string name ; 
    float rmin ; 
    float rmax ; 
};


#ifdef FUTURE
/*
This autoconv doesnt work, maybe the 
version of the json with ygltf is behind a bit ?
*/

namespace ns 
{
// a simple struct to model a person
struct person {
    std::string name;
    std::string address;
    int age;
};

void to_json(json& j, const person& p) {
    j = json{{"name", p.name}, {"address", p.address}, {"age", p.age}};
}

void from_json(const json& j, person& p) {
    p.name = j.at("name").get<std::string>();
    p.address = j.at("address").get<std::string>();
    p.age = j.at("age").get<int>();
}

} // namespace ns


void test_autoconv()
{
    ns::person p {"Ned Flanders", "744 Evergreen Terrace", 60};
}
#endif



void test_create()
{
    json j = {} ; 
    
    j["name"] = "hello" ;  
    j["fvalue"] = 1.123 ;
    j["ivalue"] = 123 ;
    j["avalue"] = {1,2,3 } ;

    std::cout << "j\n" <<  j << std::endl ; 

    json j2 = {
      {"pi", 3.141},
      {"happy", true},
      {"name", "Niels"},
      {"nothing", nullptr},
      {"answer", {
        {"everything", 42}
      }},
      {"list", {1, 0, 2}},
      {"object", {
        {"currency", "USD"},
        {"value", 42.99}
      }}
    };

    std::cout << "j2\n" << std::setw(4) <<   j2 << std::endl ; 
}


void test_append()
{
    json a = {} ; 
    a["name"] = "hello" ;  
    a["red"] = {1,2,3} ;  
    a["green"] = {4,5,6} ;  
    a["blue"] = {7,8,9} ;  
 
    json b = {} ; 
    b["cyan"] = 10 ;  
    b["magenta"] = 20 ;  
    b["yellow"] = 30 ;  

    for (const auto &j : json::iterator_wrapper(b)) a[j.key()] = j.value();

    std::cout << "a\n" << std::setw(4) << a << std::endl ; 

}


int main(int argc, char** argv)
{
    /**
    test_create();
    **/
    test_append();

    return 0 ; 
}
