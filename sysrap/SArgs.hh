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

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <algorithm>
#include <cstring>

/**
struct SArgs
==============

Allows combining standard arguments with arguments 
from a split string.

TODO: Why is this implemented in the header ? Was is for some logging reason ?
      Investigate and fix as it forces recompilation of  everything following changes.


**/


struct SArgs
{
    int argc ;
    char** argv ; 

    std::vector<std::string> elem ; 

    void add(int argc_, char** argv_)
    {
        if(argc_ == 0)  // needed for Opticks(0, NULL, argforce)  
        {
            elem.push_back("SArgsDummyExecutable"); 
        }
        else
        {
            for(int i=0 ; i < argc_ ; i++) elem.push_back(argv_[i]) ;
        }
    }

    static bool starts_with( const std::string& e, const char* pfx )
    {
        return strncmp(e.c_str(), pfx, strlen(pfx)) == 0 ;
    }

    void addElements(const std::string& line, bool dedupe)
    {
        // split string on whitespace
        std::stringstream ss(line);
        typedef std::istream_iterator<std::string> ISI ; 
        ISI begin(ss);
        ISI end ; 
        std::vector<std::string> vs(begin, end);

        for(std::vector<std::string>::const_iterator it=vs.begin() ; it != vs.end() ; it++)
        {
            std::string e = *it ; 

            /*
            Need to only skip when looks like an option, starting with "--"
            otherwise will skip repeated argument values too ... 
            which causes many tboolean- tests to fail, from skipped values
            causing option parse errors.
            */

            bool skip = dedupe && starts_with(e, "--") && std::find(elem.begin(), elem.end(), e) != elem.end() ;
            if(skip) 
            {
                printf("SArgs.hh: dedupe skipping %s \n", e.c_str());
            }
            else 
            {
                elem.push_back(e);
            }
        }
    } 

    void make()
    {
        argc = elem.size();
        argv = new char*[argc];
        for(int i=0 ; i < argc ; i++) argv[i] = const_cast<char*>(elem[i].c_str()) ;
    } 


    void dump() const 
    {
        for(int i=0 ; i < argc ; i++)
        {
            std::cout << std::setw(3) << i 
                      << std::setw(15) << elem[i]
                      << std::setw(15) << argv[i]
                      << std::endl ; 
        }
    }


    SArgs(int argc_, char** argv_, const char* extra, bool dedupe=true)
    {
        // combine standard arguments with elements split from extra 
        std::string line = extra ? extra : "" ;  
        add(argc_, argv_);
        addElements(line, dedupe);
        make();
    }

    SArgs(const char* argv0, const char* argline)
    {
        // construct standard arguments from argv0 (executable path)
        // and argline space delimited string
        std::stringstream ss ;  
        ss << argv0 << " " << argline ; 
        std::string line = ss.str() ; 
        addElements(line, false);
        make();
    }

    bool hasArg(const char* arg) const 
    {
        for(int i=0 ; i < argc ; i++) if(strcmp(argv[i], arg) == 0) return true ; 
        return false ;         
    }

    std::string getArgLine() const 
    {
        std::stringstream ss ;  
        for(int i=0 ; i < argc ; i++) ss << argv[i] << " " ; 
        return ss.str();
    }


    const char* get_arg_after(const char* option, const char* fallback) const
    {
        for(int i=1 ; i < int(elem.size()) - 1 ; i++ ) 
        {
            const char* a0 = elem[i].c_str() ; 
            const char* a1 = elem[i+1].c_str() ;
            if(a0 && strcmp(a0, option) == 0) return a1 ;   
        }
        return fallback ; 
    }

    const char* get_first_arg_ending_with(const char* ending, const char* fallback) const
    {
        for(int i=1 ; i < int(elem.size())  ; i++ ) 
        {
            const char* arg = elem[i].c_str() ; 
            int pos = strlen(arg) - strlen(ending) ;
            if( pos > 0 && strncmp(arg + pos, ending, strlen(ending)) == 0 ) return arg ;
        }
        return fallback ; 
    }


};


