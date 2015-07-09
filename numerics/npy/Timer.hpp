#pragma once

#include <vector>
#include <string>
#include <map>


class Times ; 

class Timer {
    public:
        typedef typename std::pair<std::string, double>  SD ; 
        typedef typename std::vector<SD>                VSD ; 
        typedef typename std::vector<std::string>       VS ; 
        typedef typename VSD::const_iterator            VSDI ; 
    public:
        static const char* START ; 
        static const char* STOP  ; 
    public:
        Timer();
    public:
        void start();
        void setVerbose(bool verbose);
        void setCommandLine(const std::string& cmdline);
    public:
        std::vector<std::string>& getLines();
        Times* getTimes();
    public:
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
        std::string m_commandline ; 
        Times* m_times ; 

};



inline Timer::Timer() : m_verbose(false), m_times(NULL)
{
}

inline std::vector<std::string>& Timer::getStats()
{
    return m_lines ; 
}
inline Times* Timer::getTimes()
{
    return m_times ; 
}

inline void Timer::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}
inline void Timer::setCommandLine(const std::string& cmdline)
{
    m_commandline = cmdline ; 
}

inline std::vector<std::string>& Timer::getLines()
{
    if(m_lines.size() == 0) prepTable();
    return m_lines ; 
}

