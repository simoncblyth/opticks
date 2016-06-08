#pragma once

#include <iomanip>
#include <cstring>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


class BLog {
    public:
         BLog(const char* logname="ggeoview.log", const char* loglevel="info");
         void configure(int argc, char** argv, const char* idpath=NULL);
         void init(const char* idpath);
    private:
         const char* m_logname ; 
         const char* m_loglevel ; 
};

inline BLog::BLog(const char* logname, const char* loglevel)
   :
     m_logname(strdup(logname)),
     m_loglevel(strdup(loglevel))
{
}


