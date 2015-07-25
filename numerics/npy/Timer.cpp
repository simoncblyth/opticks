#include "Timer.hpp"

#include "Times.hpp"
#include "timeutil.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


const char* Timer::START = "START" ;
const char* Timer::STOP  = "STOP" ;

void Timer::operator()(const char* mark)
{
    m_marks.push_back(SD(mark, getRealTime()));
    if(m_verbose) LOG(info) << m_name << " " << mark ; 
}

void Timer::start()
{
   (*this)(START);
}
void Timer::stop()
{
   (*this)(STOP);
   prepTable();
}

void Timer::dump(const char* msg)
{
   std::cout << m_name << " " << msg << std::endl ; 
   typedef std::vector<std::string>::const_iterator VSI ;  
   for(VSI it=m_lines.begin() ; it != m_lines.end() ; it++) std::cout << *it << std::endl ;  
}

void Timer::prepTable()
{
    double t0(0.);
    double tp(0.);

    m_times = new Times ; 

    m_lines.clear();
    if(!m_commandline.empty()) m_lines.push_back(m_commandline);

    for(VSDI it=m_marks.begin() ; it != m_marks.end() ; it++)
    {
        std::string mark = it->first ; 
        double         t = it->second ; 

        bool start = strcmp(mark.c_str(), START)==0 ;
        bool stop  = strcmp(mark.c_str(), STOP)==0 ;

        if(start) t0 = t ; 
        
        double dp = t - tp ; 
        double d0 = t - t0 ; 

        std::stringstream ss ;  

        if(!start && !stop)
        {
           ss 
             << std::fixed
             << std::setw(15) << std::setprecision(3) << d0  
             << std::setw(15) << std::setprecision(3) << dp   
             << " : " 
             << it->first   ;
        
           m_times->add(it->first.c_str(), dp);
           m_lines.push_back(ss.str());
        }

        tp = t ; 
    }
}




