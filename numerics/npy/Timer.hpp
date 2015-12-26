#pragma once

#include <vector>
#include <string>
#include <cstring>
#include <map>

class Times ; 
class TimesTable ; 

class Timer {
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
        void stop();
    public:
        void setVerbose(bool verbose);
        TimesTable* makeTable();
        static TimesTable* loadTable(const char* dir);
    public:
        const char*               getName();
    public:
        void dump(const char* msg="Timer::dump");
    private:
        VSD         m_marks ;  
        const char* m_name ; 
        bool        m_verbose ; 

};

inline Timer::Timer(const char* name) 
       : 
       m_name(strdup(name)),
       m_verbose(false)
{
}

inline void Timer::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}
inline const char* Timer::getName()
{
    return m_name ; 
}
