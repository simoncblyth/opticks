#include "NTxt.hpp"
#include "BLog.hh"
#include <iostream>
#include <fstream>


void NTxt::read()
{
   std::ifstream in(m_path, std::ios::in);
    if(!in.is_open()) 
    {   
        LOG(fatal) << "NTxt::read failed to open " << m_path ; 
        return ;
    }   

    std::string line ; 
    while(std::getline(in, line))
    {   
         m_lines.push_back(line);
    }   
    in.close();

    LOG(info) << "NTxt::read " 
              << " path " << m_path 
              << " lines " << m_lines.size() 
              ;   

}
