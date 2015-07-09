#include "Report.hpp"

#include "jsonutil.hpp"
#include "timeutil.hpp"

#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>

#include <sstream>


#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


const char* Report::TIMEFORMAT = "%Y%m%d_%H%M%S" ;

void Report::save(const char* dir, const char* name)
{
    std::string path = preparePath(dir, name, true);
    if(path.empty()) return ; 

    LOG(info)<<"Report::save to " << path ; 
    std::ofstream ofs(path) ;
    std::copy( m_lines.begin(), m_lines.end(), std::ostream_iterator<std::string>(ofs, "\n"));

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

