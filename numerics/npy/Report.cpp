#include "Report.hpp"

#include "jsonutil.hh"
#include "timeutil.hpp"

#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>

#include <sstream>

#include "BLog.hh"

const char* Report::NAME = "report.txt" ;
const char* Report::TIMEFORMAT = "%Y%m%d_%H%M%S" ;

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
    std::string path = preparePath(dir, name, true);
    if(path.empty()) return ; 

    LOG(info)<<"Report::load from " << path ; 
    std::ifstream ifs(path) ;
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
    std::string path = preparePath(dir, name, true);
    if(path.empty()) return ; 

    LOG(info)<<"Report::save to " << path ; 
    std::ofstream ofs(path) ;
    std::copy( m_lines.begin(), m_lines.end(), std::ostream_iterator<std::string>(ofs, "\n"));

}

void Report::dump(const char* msg)
{
    LOG(info) << msg ; 
    for(unsigned int i=0 ; i < m_lines.size() ; i++ ) std::cout << m_lines[i] << std::endl ; 
}



std::string Report::timestamp()
{
    char* tsl =  now(Report::TIMEFORMAT, 20, 0);
    std::string timestamp =  tsl ;
    free((void*)tsl);
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

