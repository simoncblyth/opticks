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

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <string>


int main(int, char** argv)
{
    const char* flags = "$TMP/GFlagIndexLocal.ini";
    std::ifstream fs(flags, std::ios::in);

    if(!fs.is_open())
    {
         std::cout << argv[0] << " " << "missing input file " << flags << std::endl ;
         return 0 ; 
    } 

    std::string line = ""; 

    std::vector<std::string> names ; 
    std::vector<unsigned int> vals ; 

    while(!fs.eof()) 
    {   
        std::getline(fs, line);
        
        const char* eq = strchr( line.c_str(), '=');
        if(eq)
        {
            std::string name = line.substr(0, eq - line.c_str());        
            std::string value = eq + 1 ;        
    
            names.push_back(name);
            vals.push_back(atoi(value.c_str()));
 
            std::cout 
                << std::setw(30) << name 
                << std::setw(30) << value 
                << std::endl ; 
        }
    }   


    for(unsigned int i=0 ; i<names.size() ; i++)
    {
        std::cout 
              << std::setw(30) << i
              << std::setw(30) << names[i] 
              << std::setw(30) << vals[i] 
              << std::endl ; 
    }

    return 0 ;
}



