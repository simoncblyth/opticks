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


**/


struct SArgs
{
    int argc ;
    char** argv ; 

    std::vector<std::string> elem ; 

    void add(int argc_, char** argv_)
    {
        for(int i=0 ; i < argc_ ; i++) elem.push_back(argv_[i]) ;
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
            bool skip = dedupe && std::find(elem.begin(), elem.end(), e) != elem.end() ;
            if(skip) 
                printf("dedupe skipping %s \n", e.c_str());
            else 
                elem.push_back(e);
        }
    } 

    void make()
    {
        argc = elem.size();
        argv = new char*[argc];
        for(int i=0 ; i < argc ; i++) argv[i] = const_cast<char*>(elem[i].c_str()) ;
    } 


    void dump()
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

    bool hasArg(const char* arg)
    {
        for(int i=0 ; i < argc ; i++) if(strcmp(argv[i], arg) == 0) return true ; 
        return false ;         
    }

    std::string getArgLine()
    {
        std::stringstream ss ;  
        for(int i=0 ; i < argc ; i++) ss << argv[i] << " " ; 
        return ss.str();
    }


};


