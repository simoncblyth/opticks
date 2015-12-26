#include "Timer.hpp"

#include "Times.hpp"
#include "TimesTable.hpp"

#include "timeutil.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

#include "NLog.hpp"

const char* Timer::COLUMNS = "t_absolute,t_delta" ;
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
}


void Timer::dump(const char* msg)
{
    TimesTable* tt = makeTable();
    tt->dump(msg);
}


TimesTable* Timer::loadTable(const char* dir)
{
    TimesTable* tt = new TimesTable(COLUMNS) ; 
    tt->load(dir);
    return tt ;
}

TimesTable* Timer::makeTable()
{
    TimesTable* tt = new TimesTable(COLUMNS) ; 

    double t0(0.);
    double tp(0.);

    for(VSDI it=m_marks.begin() ; it != m_marks.end() ; it++)
    {
        std::string mark = it->first ; 
        double         t = it->second ; 

        bool start = strcmp(mark.c_str(), START)==0 ;
        bool stop  = strcmp(mark.c_str(), STOP)==0 ;

        if(start) t0 = t ; 
        
        double d0 = t - t0 ; 
        double dp = t - tp ; 

        if(!start && !stop)
        {
           tt->getColumn(0)->add(it->first.c_str(), d0);
           tt->getColumn(1)->add(it->first.c_str(), dp);
        }
        tp = t ; 
    }
    return tt ;
}


