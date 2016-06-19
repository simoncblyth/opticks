#include <cassert>
#include <sstream>
#include <iostream>
#include <iomanip>

// brap-
#include "BFile.hh"
#include "BList.hh"
#include "BLog.hh"

#include "Times.hpp"





Times::Times(const char* label)  
   : 
     m_scale(1.0) , 
     m_label(strdup(label)) 
{
}

void Times::setLabel(const char* label)
{
    m_label = strdup(label);
}

Times* Times::clone(const char* label)
{
    Times* ts = new Times(label) ; 
    for(VSD::const_iterator it=m_times.begin() ; it != m_times.end() ; it++) ts->add(it->first.c_str(), it->second) ;
    return ts ; 
}

unsigned int Times::getNumEntries()
{
    return m_times.size();
}
std::pair<std::string, double>&  Times::getEntry(unsigned int i)
{
    return m_times[i] ;
}

void Times::add(const char* name, double t )
{
    m_times.push_back(SD(name, t));
}
unsigned int Times::getSize()
{
    return m_times.size();
}
std::vector<std::pair<std::string, double> >& Times::getTimes()
{
    return m_times ;
}
double Times::getScale()
{
    return m_scale ; 
}
void Times::setScale(double scale)
{
    m_scale = scale  ; 
}


const char* Times::getLabel()
{
    return m_label ; 
}





void Times::save(const char* dir)
{
    std::string nam = name();
    std::string path = BFile::preparePath(dir, nam.c_str(), true);
    LOG(info) << "Times::save to " << path ;
    BList<std::string, double>::save( &m_times, dir, nam.c_str());
}


Times* Times::load(const char* label, const char* dir, const char* name)
{
    Times* t = new Times(label) ;
    t->load(dir, name);
    return t ; 
}
void Times::load(const char* dir, const char* name)
{
    BList<std::string, double>::load( &m_times, dir, name);
}
void Times::load(const char* dir)
{
    std::string nam = name();
    load(dir, nam.c_str()) ; 
}

void Times::dump(const char* msg)
{
   LOG(info) << msg ; 
   for(VSD::const_iterator it=m_times.begin() ; it != m_times.end() ; it++)
   {
       std::cout 
          <<  std::setw(25) << it->first 
          <<  std::setw(25) << it->second
          <<  std::endl ; 
   } 
}

std::string Times::name()
{
    std::stringstream ss ; 
    ss << m_label << ".ini" ;
    return ss.str();
}

std::string Times::name(const char* typ, const char* tag)
{
    std::stringstream ss ; 
    ss << typ << "_" << tag << ".ini" ;
    return ss.str();
}

void Times::compare(const std::vector<Times*>& vt, unsigned int nwid, unsigned int twid, unsigned int tprec)
{
    unsigned int n = vt.size();

    // check are all same size : ie same number of timings
    unsigned int size = 0 ; 
    for(unsigned int i=0 ; i < n ; i++)
    {
        if(i == 0) 
              size = vt[i]->getSize() ;
        else  
              assert( vt[i]->getSize() == size ); 
    }


    // line for the label
    std::cout << std::setw(nwid) << "label" ;
    for(unsigned int i=0 ; i < n ; i++)
    {
        const char* label = vt[i]->getLabel() ; 
        std::cout << std::setw(twid) << ( label ? label : "" ) ;
    }
    std::cout << std::endl ; 



    // line for the scale
    std::cout << std::setw(nwid) << "scale" ;
    for(unsigned int i=0 ; i < n ; i++)
    {
        double s = vt[i]->getScale() ; 
        std::cout << std::setw(twid) << std::setprecision(tprec) << std::fixed << s ;
    }
    std::cout << std::endl ; 



    // over those timings
    for(unsigned int j=0 ; j < size ; j++ )
    {
         // check all aligned with same name
         std::string name ; 
         for(unsigned int i=0 ; i < n ; i++)
         {
             std::string iname = vt[i]->getTimes()[j].first ; 
             if(name.empty()) 
                  name = iname ;
             else
                   assert(strcmp(name.c_str(), iname.c_str())==0);
         } 
         std::cout << std::setw(nwid) << name ;

         for(unsigned int i=0 ; i < n ; i++)
         {
             double t = vt[i]->getTimes()[j].second ; 
             double s = vt[i]->getScale() ; 
             std::cout << std::setw(twid) << std::setprecision(tprec) << std::fixed << t*s ;
         }
         std::cout << std::endl ; 
    }
}


