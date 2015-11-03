#pragma once

#include <cstring>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

class NLog {
    public:
         NLog(const char* logname="ggeoview.log", const char* loglevel="info");
         void configure(int argc, char** argv);
         void init(const char* idpath);
    private:
         const char* m_logname ; 
         const char* m_loglevel ; 
};

inline NLog::NLog(const char* logname, const char* loglevel)
   :
     m_logname(strdup(logname)),
     m_loglevel(strdup(loglevel))
{
}


