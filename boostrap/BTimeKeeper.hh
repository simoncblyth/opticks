#pragma once

#include <vector>
#include <string>
#include <cstring>
#include <map>

class BTimesTable ; 

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

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

