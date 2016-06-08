#pragma once

#include <iomanip>
#include <cstring>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


class BLog {
    public:
         BLog(const char* logname="ggeoview.log", const char* loglevel="info");
         virtual ~BLog();
         void configure(int argc, char** argv, const char* idpath=NULL);
         void init(const char* idpath);
         void setPause(bool pause=true);
         void setExitPause(bool pause=true);
    private:
         const char* m_logname ; 
         const char* m_loglevel ; 
         bool        m_pause ;    
         bool        m_exitpause ;    
};

inline BLog::BLog(const char* logname, const char* loglevel)
   :
     m_logname(strdup(logname)),
     m_loglevel(strdup(loglevel)),
     m_pause(false),
     m_exitpause(false)
{
}

inline void BLog::setPause(bool pause)
{
    m_pause = pause ; 
}


inline void BLog::setExitPause(bool pause)
{
    m_exitpause = pause ; 
}

