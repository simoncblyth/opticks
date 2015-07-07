#pragma once

#include <vector>
#include <string>
#include <map>

class Timer {
    public:
        typedef typename std::pair<std::string, double>  SD ; 
        typedef typename std::vector<SD>                VSD ; 
        typedef typename std::vector<std::string>       VS ; 
        typedef typename VSD::const_iterator            VSDI ; 
    public:
        static const char* START ; 
        static const char* STOP  ; 
        Timer();
    public:
        void start();
        void setVerbose(bool verbose);
        void operator()(const char* mark);
        void stop();
    private:
        void prepTable();
    public:
        void dump(const char* msg="Timer::dump");
        std::vector<std::string>& getStats();
    private:
        VSD  m_marks ;  
        VS   m_lines ; 
        bool m_verbose ; 
};



inline Timer::Timer() : m_verbose(false)
{
}

inline std::vector<std::string>& Timer::getStats()
{
    return m_lines ; 
}

inline void Timer::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}

