#pragma once

#include <iomanip>
#include <cstring>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

class BLog {
    public:
         BLog(int argc, char** argv);
         void setDir( const char* dir);
         virtual ~BLog();
    private:
         void init();
         void configure(int argc, char** argv);
         void setName( const char* logname);
         void setLevel( const char* loglevel);
         void setup( const char* level_);
         void addFileLog();
    private:
         void setPause(bool pause=true);
         void setExitPause(bool pause=true);
    private:
         int         m_argc ; 
         char**      m_argv ; 

         const char* m_logname ; 
         const char* m_loglevel ; 
         const char* m_logdir ; 

         bool        m_pause ;    
         bool        m_exitpause ;    
         bool        m_addfile ; 
         bool        m_setup ; 
};

inline BLog::BLog(int argc, char** argv)
   :
     m_argc(argc),
     m_argv(argv),

     m_logname(NULL),
     m_loglevel(NULL),
     m_logdir(NULL),

     m_pause(false),
     m_exitpause(false),
     m_addfile(false),
     m_setup(false)
{
     init();
}

inline void BLog::setPause(bool pause)
{
    m_pause = pause ; 
}
inline void BLog::setExitPause(bool pause)
{
    m_exitpause = pause ; 
}
inline void BLog::setName(const char* logname)
{
    if(logname) m_logname = strdup(logname) ;
}
inline void BLog::setLevel(const char* loglevel)
{
    if(loglevel) m_loglevel = strdup(loglevel) ;
}


