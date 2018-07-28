#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <climits>

#include "BFile.hh"

#include "NTxt.hpp"
#include "PLOG.hh"

#ifdef _MSC_VER
#define strdup _strdup
#endif



NTxt::NTxt(const char* path)
   :
   m_path(strdup(path))
{
}

std::string NTxt::desc() const 
{  
    std::stringstream ss ;
    ss << "NTxt " 
       << " path " << m_path  
       << " NumLines " << getNumLines()
       ;
    return ss.str();
}


const char* NTxt::getLine(unsigned int num) const 
{
   return num < m_lines.size() ? m_lines[num].c_str() : NULL ; 
}
unsigned int  NTxt::getNumLines() const 
{
   return m_lines.size() ; 
}

void NTxt::dump(const char* msg) const 
{
    unsigned n = getNumLines() ;
    LOG(info) << msg << " NumLines " << n ; 
    for(unsigned i=0 ; i < n ; i++) std::cout << getLine(i) << std::endl ; 
}


unsigned int NTxt::getIndex(const char* line) const 
{
   std::string s(line);
   for(unsigned int i=0 ; i < m_lines.size() ; i++) if(m_lines[i].compare(s)==0) return i ;
   return UINT_MAX ; 
}


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

    LOG(debug) << "NTxt::read " 
              << " path " << m_path 
              << " lines " << m_lines.size() 
              ;   

}


void NTxt::prepDir() const 
{
    std::string pdir = BFile::ParentDir(m_path);
    BFile::CreateDir(pdir.c_str()); 

    LOG(debug) << "NTxt::prepDir"
              << " pdir " << pdir
              ;
}

void NTxt::write() const 
{
    prepDir();

    std::ofstream out(m_path, std::ios::out);
    if(!out.is_open()) 
    {   
        LOG(fatal) << "NTxt::write failed to open " << m_path ; 
        return ;
    }   


    for(VS_t::const_iterator it=m_lines.begin() ; it != m_lines.end() ; it++)
    {
        out << *it << std::endl ; 
    }

    out.close();
}

void NTxt::addLine(const std::string& line)
{
    addLine(line.c_str());
}
void NTxt::addLine(const char* line)
{
    m_lines.push_back(line);
}

