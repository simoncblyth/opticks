#include "BTimeKeeper.hh"

#include "BTimes.hh"
#include "BTimesTable.hh"
#include "BTimeStamp.hh"

#include <iostream>
#include <iomanip>
#include <sstream>

#include "PLOG.hh"

const char* BTimeKeeper::COLUMNS = "t_absolute,t_delta" ;
const char* BTimeKeeper::START = "START" ;
const char* BTimeKeeper::STOP  = "STOP" ;


BTimeKeeper::BTimeKeeper(const char* name) 
    : 
    m_name(strdup(name)),
    m_verbose(false)
{
}

void BTimeKeeper::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}
const char* BTimeKeeper::getName()
{
    return m_name ; 
}

void BTimeKeeper::stamp(const char* mark)
{
    m_marks.push_back(SD(mark, BTimeStamp::RealTime() ));
    if(m_verbose) 
    {
       LOG(debug) << m_name << " " << mark ; 
    }
}

void BTimeKeeper::operator()(const char* mark)
{
    stamp(mark);
}
void BTimeKeeper::start()
{
   (*this)(START);
}
void BTimeKeeper::stop()
{
   (*this)(STOP);
}


double BTimeKeeper::deltaTime(int i0, int i1) const 
{
    int num = m_marks.size() ;
    if(num < 2) return -999.0 ; 
    unsigned j0 = i0 < 0 ? i0+num : i0 ;  
    unsigned j1 = i1 < 0 ? i1+num : i1 ;  
    const SD& m0 = m_marks[j0] ;   
    const SD& m1 = m_marks[j1] ;   
    double dt = m1.second - m0.second ;  
    return dt ; 
}



void BTimeKeeper::dump(const char* msg)
{
    double dt = deltaTime() ; 
    LOG(info) << "deltaTime " << dt ; 

    BTimesTable* tt = makeTable();
    tt->dump(msg);
}


BTimesTable* BTimeKeeper::loadTable(const char* dir)
{
    BTimesTable* tt = new BTimesTable(COLUMNS) ; 
    tt->load(dir);
    return tt ;
}

BTimesTable* BTimeKeeper::makeTable()
{
    BTimesTable* tt = new BTimesTable(COLUMNS) ; 

    double t0(0.);
    double tp(0.);

    for(VSDI it=m_marks.begin() ; it != m_marks.end() ; it++)
    {
        const std::string& mark = it->first ; 
        double         t = it->second ; 

        const char* mk = mark.c_str() ; 

        bool start_ = strcmp(mk, START)==0 ;
        bool stop_  = strcmp(mk, STOP)==0 ;

        if(start_) t0 = t ; 
        
        double d0 = t - t0 ;   // time since start 
        double dp = t - tp ;   // time since last mark

        if(!start_ && !stop_)
        {
           tt->getColumn(0)->add(mk, d0);
           tt->getColumn(1)->add(mk, dp);
        }
        tp = t ; 
    }
    return tt ;
}


