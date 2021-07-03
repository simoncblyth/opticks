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

#include "SStr.hh"
#include "BDir.hh"

#include <cstring>
#include <string>
#include <iostream>
#include <boost/filesystem.hpp>


namespace fs = boost::filesystem;

void BDir::dirlist(std::vector<std::string>& basenames,  const char* path, const char* ext)
{
    fs::path dir(path);
    if(!( fs::exists(dir) && fs::is_directory(dir))) return ; 

    fs::directory_iterator it(dir) ;
    fs::directory_iterator end ;
   
    for(; it != end ; ++it)
    {
        std::string fname = it->path().filename().string() ;
        const char* fnam = fname.c_str();

        if(strlen(fnam) > strlen(ext) && strcmp(fnam + strlen(fnam) - strlen(ext), ext)==0)
        {
            std::string basename(fnam, fnam+strlen(fnam)-strlen(ext));
            //std::cout << basename << std::endl ;
            basenames.push_back(basename);
        }
    }
}


void BDir::dirlist(std::vector<std::string>& names,  const char* path)
{
    fs::path dir(path);
    if(!( fs::exists(dir) && fs::is_directory(dir))) return ; 

    fs::directory_iterator it(dir) ;
    fs::directory_iterator end ;
   
    for(; it != end ; ++it)
    {
        std::string name = it->path().filename().string() ;
        //std::cout << name << std::endl ;
        names.push_back(name);
    }
}

/**
BDir::dirdirlist
------------------

Collect the *names* of subdirectories within the *path* directory.
When *name_suffix* is non-null and *endswith* is true only names with the 
provided suffix are collected. When *endswith* is false only names **NOT** ending 
with the suffix are collected.   

**/

void BDir::dirdirlist(std::vector<std::string>& names,  const char* path, const char* name_suffix, bool endswith )
{
    fs::path dir(path);
    if(!( fs::exists(dir) && fs::is_directory(dir))) return ; 

    fs::directory_iterator it(dir) ;
    fs::directory_iterator end ;
   
    for(; it != end ; ++it)
    {
        if(fs::is_directory(it->path()))
        {
            std::string name = it->path().filename().string() ;
            //std::cout << "dirdirlist " << name << std::endl ;
            bool select = name_suffix == nullptr ? true : SStr::EndsWith(name.c_str(), name_suffix ) == endswith  ; 
            if(select)  names.push_back(name);
        }
    }
}


void BDir::dirdirlist(std::vector<std::string>& names,  const char* path )
{
    BDir::dirdirlist(names, path, nullptr, false ); 
}

