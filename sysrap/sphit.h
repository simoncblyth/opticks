#pragma once
/**
sphit.h
=========

Used by SEvt::getLocalHit interface for 
passing information from sframe into U4Hit 

Currently node_index is not included. As node_index is primarily 
of interest for debugging, it is not so critical to expose it all 
the way up to U4Hit level. 

Also stree.h has inst_nidx which will make getting the nidx 
of each instance straightforward : so node_index inclusion 
can wait until stree is more firmly integrated.

**/

#include <string>
#include <sstream>
#include <iomanip>

struct sphit
{
    int iindex ; 
    int sensor_identifier ; 
    int sensor_index ; 
    // int node_index ;   

    void zero(); 
    std::string desc() const ; 

}; 

inline void sphit::zero()
{
    iindex = 0 ; 
    sensor_identifier = 0 ; 
    sensor_index = 0 ; 
}

inline std::string sphit::desc() const 
{
    std::stringstream ss ; 
    ss << "sphit::desc"
       << " iindex " << std::setw(7) << iindex 
       << " sensor_identifier " << std::setw(7) << sensor_identifier 
       << " sensor_index " << std::setw(7) << sensor_index 
       ;
    std::string s = ss.str(); 
    return s ; 
}


