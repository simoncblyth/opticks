#pragma once

#include <vector>
#include <string>
#include <cstring>
#include <map>

class BTimesTable ; 

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

/**
BTimeKeeper
================

m_marks
    vector of string double pairs 


Instances of BTimeKeeper, typically m_timer are held by several 
objects:: 

    [blyth@localhost opticks]$ opticks-f new\ BTimeKeeper
    ./boostrap/BTimeKeeper.hh:     574     m_timer = new BTimeKeeper("Opticks::");
    ./npy/NDualContouringSample.cpp:   m_timer(new BTimeKeeper),
    ./npy/NImplicitMesher.cpp:    m_timer(new BTimeKeeper),
    ./optickscore/Opticks.cc:    m_timer = new BTimeKeeper("Opticks::");
    ./optickscore/OpticksEvent.cc:    m_timer = new BTimeKeeper("OpticksEvent"); 
    ./optixrap/OScene.cc:    m_timer(new BTimeKeeper("OScene::")),

::

     568 void Opticks::init()
     569 {
     ...
     574     m_timer = new BTimeKeeper("Opticks::");
     575 
     576     m_timer->setVerbose(true);
     577 
     578     m_timer->start();



**/


class BRAP_API BTimeKeeper {
    public:
        static const char* COLUMNS ; 
    public:
        typedef std::pair<std::string, double>  SD ; 
        typedef std::vector<SD>                VSD ; 
        typedef std::vector<std::string>       VS ; 
        typedef VSD::const_iterator            VSDI ; 
    public:
        static const char* START ; 
        static const char* STOP  ; 
    public:
        BTimeKeeper(const char* name="");
    public:
        void start();
        void operator()(const char* mark);
        void stamp(const char* mark);
        void stop();
    public:
        void setVerbose(bool verbose);
        BTimesTable*        makeTable();
        static BTimesTable* loadTable(const char* dir);
    public:
        const char*               getName();
        double deltaTime(int i0=0, int i1=-1) const ;
    public:
        void dump(const char* msg="BTimeKeeper::dump");
    private:
        VSD         m_marks ;  
        const char* m_name ; 
        bool        m_verbose ; 

};

#include "BRAP_TAIL.hh"

