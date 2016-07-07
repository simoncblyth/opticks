#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>

// brap-
#include "BTime.hh"
#include "BFile.hh"

// npy-
#include "Report.hpp"

#include "PLOG.hh"

const char* Report::NAME = "report.txt" ;
const char* Report::TIMEFORMAT = "%Y%m%d_%H%M%S" ;


Report::Report()
{
}

void Report::add(const VS& lines)
{
    for(VS::const_iterator it=lines.begin() ; it!=lines.end() ; it++) m_lines.push_back(*it);
}



void Report::save(const char* dir)
{
    save(dir, NAME);
}

Report* Report::load(const char* dir)
{
    Report* report = new Report ; 
    report->load(dir, NAME);
    return report ; 
}

void Report::load(const char* dir, const char* name)
{
    std::string path = BFile::preparePath(dir, name, true);
    if(path.empty()) return ; 

    LOG(info)<<"Report::load from " << path ; 
    std::ifstream ifs(path.c_str()) ;
    //std::copy(std::istream_iterator<std::string>(ifs), std::istream_iterator<std::string>(), std::back_inserter(m_lines));
    std::string line;
    while ( std::getline(ifs, line) )
    {
        if (line.empty()) continue;
        m_lines.push_back(line);
    }
}

void Report::save(const char* dir, const char* name)
{
    std::string path = BFile::preparePath(dir, name, true);
    if(path.empty()) return ; 

    LOG(info)<<"Report::save to " << path ; 
    std::ofstream ofs(path.c_str()) ;
    std::copy( m_lines.begin(), m_lines.end(), std::ostream_iterator<std::string>(ofs, "\n"));

}

void Report::dump(const char* msg)
{
    LOG(info) << msg ; 
    for(unsigned int i=0 ; i < m_lines.size() ; i++ ) std::cout << m_lines[i] << std::endl ; 
}


std::string Report::timestamp()
{
    std::string timestamp =  BTime::now(Report::TIMEFORMAT,0);
    return timestamp ; 
}

std::string Report::name(const char* typ, const char* tag)
{
    std::stringstream ss ; 
    ss << Report::timestamp() 
       << "_"
       << typ
       << "_"
       << tag
       << ".txt"
       ;

    return ss.str() ;
}


