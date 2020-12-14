/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

// formerly npy/Times.cpp

#include <cassert>
#include <sstream>
#include <iostream>
#include <sstream>
#include <iomanip>

// brap-
#include "BStr.hh"
#include "BFile.hh"
#include "BList.hh"
#include "BTimes.hh"

#include "PLOG.hh"


BTimes::BTimes(const char* label)  
    : 
    m_scale(1.0) , 
    m_label(strdup(label)) 
{
}

BTimes::~BTimes()
{
    free((char*)m_label); 
}


void BTimes::setLabel(const char* label)
{
    m_label = strdup(label);
}

BTimes* BTimes::clone(const char* label)
{
    BTimes* ts = new BTimes(label) ; 
    for(VSD::const_iterator it=m_times.begin() ; it != m_times.end() ; it++) ts->add(it->first.c_str(), it->second) ;
    return ts ; 
}

unsigned int BTimes::getNumEntries()
{
    return m_times.size();
}
std::pair<std::string, double>&  BTimes::getEntry(unsigned int i)
{
    return m_times[i] ;
}


/**
BTimes::add
---------------

Collects into m_times vector of (string, double) pairs.

**/


void BTimes::add(const char* name_, double t )
{
    m_times.push_back(SD(name_, t));
}



void BTimes::add(const char* name_, int idx, double t )
{
    std::stringstream ss ; 
    ss << name_ 
       << std::setw(3) << std::setfill('0') << idx 
       ;

    std::string s = ss.str(); 
    add( s.c_str(), t );     
}



unsigned int BTimes::getSize()
{
    return m_times.size();
}
std::vector<std::pair<std::string, double> >& BTimes::getTimes()
{
    return m_times ;
}
double BTimes::getScale()
{
    return m_scale ; 
}
void BTimes::setScale(double scale)
{
    m_scale = scale  ; 
}

const char* BTimes::getLabel()
{
    return m_label ; 
}

void BTimes::save(const char* dir)
{
    std::string nam = name();
    std::string path = BFile::preparePath(dir, nam.c_str(), true);
    LOG(debug) << "BTimes::save to " << path ;
    BList<std::string, double>::save( &m_times, dir, nam.c_str());
}


BTimes* BTimes::Load(const char* label, const char* dir, const char* name_)
{
    BTimes* t = new BTimes(label) ;
    t->load(dir, name_);
    return t ; 
}
BTimes* BTimes::Load(const char* label, const char* dir )
{
    BTimes* t = new BTimes(label) ;
    t->load(dir);
    return t ; 
}


void BTimes::load(const char* dir, const char* name_)
{
    BList<std::string, double>::load( &m_times, dir, name_);
}
void BTimes::load(const char* dir)
{
    std::string nam = name();
    load(dir, nam.c_str()) ; 
}


void BTimes::addAverage(const char* prefix )
{
   int count(0); 

   double sum(0.); 
   for(VSD::const_iterator it=m_times.begin() ; it != m_times.end() ; it++)
   {
       const std::string& name = it->first ; 
       double t = it->second ; 
       if( BStr::StartsWith(name.c_str(), prefix))
       { 
           count += 1 ;
           sum += t ;  
       } 
   } 

   double avg = count > 0 ? sum/count : sum ;   
   std::stringstream ss ;  
   ss << prefix << "AVG" ;  
   std::string s = ss.str(); 
   add( s.c_str(), avg );  
}


void BTimes::dump(const char* msg)
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

std::string BTimes::name()
{
    std::stringstream ss ; 
    ss << m_label << ".ini" ;
    return ss.str();
}

std::string BTimes::name(const char* typ, const char* tag)
{
    std::stringstream ss ; 
    ss << typ << "_" << tag << ".ini" ;
    return ss.str();
}


/*
void BTimes::compare(const BTimes* a, const BTimes* b, unsigned int nwid, unsigned int twid, unsigned int tprec) // static
{
    const std::vector<BTimes*> vt = { a, b } ;
    compare( vt, nwid, twid, tprec );
}
*/


void BTimes::compare(const std::vector<BTimes*>& vt, unsigned int nwid, unsigned int twid, unsigned int tprec) // static
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


