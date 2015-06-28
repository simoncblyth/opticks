#include "Flags.hh"

#include "assert.h"
#include "stdio.h"
#include "stdlib.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>


void Flags::read(const char* path)
{
    std::ifstream fs(path, std::ios::in);
    std::string line = ""; 
    while(!fs.eof()) 
    {   
        std::getline(fs, line);
        char* eq = strchr( line.c_str(), '=');
        if(eq)
        {
            std::string name = line.substr(0, eq - line.c_str());        
            std::string value = eq + 1 ;        
            unsigned int code = atoi(value.c_str()) ;  
            m_name2code[name] = code ; 
            m_code2name[code] = name ; 
        }
    }   
    assert(m_name2code.size() == m_code2name.size());
}


std::string Flags::getSequenceString(unsigned long long seq)
{
    unsigned int bits = sizeof(seq)*8 ; 
    unsigned int slots = bits/4 ; 

    std::stringstream ss ; 
    for(unsigned int i=0 ; i < slots ; i++)
    {
        unsigned int i4 = i*4 ; 
        unsigned long long mask = (0xFull << i4) ;   // tis essential to use 0xFull rather than 0xF for going beyond 32 bit
        unsigned long long portion = (seq & mask) >> i4 ;  
        unsigned int code = portion ;
        std::string name = m_code2name.count(code) == 1 ? m_code2name[code] : "???" ; 
     /*
        std::cout << std::setw(3) << std::dec << i 
                  << std::setw(30) << std::hex << portion 
                  << std::setw(30) << std::hex << code
                  << std::setw(30) << name   
                  << std::endl ; 
      */

        ss << name << " " ; 
    }
    return ss.str() ; 
}

void Flags::dump(const char* msg)
{
    std::cout << msg << std::endl ; 
    typedef std::map<unsigned int, std::string> MUS ; 
    for(MUS::iterator it=m_code2name.begin() ; it != m_code2name.end() ; it++)
    {    
        unsigned int code = it->first ; 

        std::cout 
              << std::setw(3) << std::dec << code
              << std::setw(3) << std::hex << code
              << std::setw(30) << it->second 
              << std::endl ; 
    }
}

