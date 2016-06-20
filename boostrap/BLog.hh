#pragma once

#include <iostream>
#include <iomanip>
#include <cstring>


// headers here so the macro can work elsewhere easily
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include "boost/log/utility/setup.hpp"





#define LOG BOOST_LOG_TRIVIAL


// trace/debug/info/warning/error/fatal

#include "BRAP_API_EXPORT.hh"
#include "BRAP_FLAGS.hh"


// textual workaround to get across the dll divide
// using static methods of Blog 
#define BLOG(argc, argv) \
{ \
    boost::log::add_common_attributes(); \
    boost::log::register_simple_formatter_factory< boost::log::trivial::severity_level, char >("Severity");  \
    boost::log::add_console_log(  \
        std::cerr,   \
        boost::log::keywords::format = "[%TimeStamp%]:%Severity%: %Message%", \
        boost::log::keywords::auto_flush = true \
    );  \
    boost::log::core::get()->set_filter \
    (  \
        boost::log::trivial::severity >= BLog::FilterLevel((argc), (argv)) \
    ); \
} \



class BRAP_API BLog {
    public:
         static int FilterLevel(int argc, char** argv);
         static int FilterLevel(const char* level);
    public:
         // smth doesnt cross the dll divide
         static void setFilter(const char* level);
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


