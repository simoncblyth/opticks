#include "Times.hpp"

#include "assert.h"
#include "jsonutil.hpp"

#include <sstream>
#include <iostream>
#include <iomanip>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void Times::save(const char* dir, const char* name)
{
    saveList<std::string, double>( m_times, dir, name);
}
void Times::load_(const char* dir, const char* name)
{
    loadList<std::string, double>( m_times, dir, name);
}
Times* Times::load(const char* dir, const char* name)
{
    Times* t = new Times ; 
    t->load_(dir, name);
    return t ; 
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



