#pragma once

#include <vector>
#include <string>
#include <cstring>
#include <map>


class Times ; 

class Timer {
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
        void setVerbose(bool verbose);
        void setCommandLine(const std::string& cmdline);
    public:
        std::vector<std::string>& getLines();
        Times*                    getTimes();
        const char*               getName();
    public:
        void operator()(const char* mark);
        void stop();
    private:
        void prepTable();
    public:
        void dump(const char* msg="Timer::dump");
        std::vector<std::string>& getStats();
    private:
        VSD         m_marks ;  
        VS          m_lines ; 
        const char* m_name ; 
        bool        m_verbose ; 
        std::string m_commandline ; 
        Times*      m_times ; 

};



inline Timer::Timer(const char* name) 
       : 
       m_name(strdup(name)),
       m_verbose(false), 
       m_times(NULL)
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
inline const char* Timer::getName()
{
    return m_name ; 
}
