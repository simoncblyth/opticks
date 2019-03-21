#pragma once

#include <vector>
#include <string>
#include <cstring>
#include <map>

class Times ; 
class TimesTable ; 

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API Timer {
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
        Timer(const char* name="");
    public:
        void start();
        void operator()(const char* mark);
        void stamp(const char* mark);
        void stop();
    public:
        void setVerbose(bool verbose);
        TimesTable* makeTable();
        static TimesTable* loadTable(const char* dir);
    public:
        const char*               getName();
        double deltaTime(int i0=0, int i1=-1) const ;
    public:
        void dump(const char* msg="Timer::dump");
    private:
        VSD         m_marks ;  
        const char* m_name ; 
        bool        m_verbose ; 

};

#include "NPY_TAIL.hh"

