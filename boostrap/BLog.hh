#pragma once

#include <iostream>
#include <iomanip>
#include <cstring>

#ifdef USE_BOOST_LOG

// http://stackoverflow.com/questions/24302123/how-can-i-use-boost-log-across-dll-boundaries
// http://www.boost.org/doc/libs/1_61_0/libs/log/doc/html/log/installation/config.html

/*
If your application consists of more than one module (e.g. an exe and one or
several dll's) that use Boost.Log, the library must be built as a shared
object.
*/
// headers here so the macro can work elsewhere easily
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include "boost/log/utility/setup.hpp"

#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


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

#else

#include <plog/Log.h>
// translate from boost log levels to plog 
using plog::fatal ;
using plog::error ;
using plog::warning ;
using plog::info ;
using plog::debug ;
using plog::verbose ;
// defines are dangerous 
#define trace plog::verbose

#endif


#include "BRAP_API_EXPORT.hh"
#include "BRAP_FLAGS.hh"

class BRAP_API BLog {
    public:
         static void Initialize(void* whatever, int level);
         static int  SeverityLevel(const char* ll);
    public:
         BLog(int argc, char** argv);
         void setDir( const char* dir);
         int getLevel();
         virtual ~BLog();
    private:
         void init();
         void parse(int argc, char** argv);
         void setName( const char* logname);
         void setLevel( const char* loglevel);
         void initialize( void* whatever );
         void addFileLog();
    private:
         int         m_argc ; 
         char**      m_argv ; 

         int         m_loglevel ; 
         const char* m_logname ; 
         const char* m_logdir ; 

         bool        m_nogeocache ; 
         bool        m_pause ;    
         bool        m_exitpause ;    
         bool        m_addfile ; 
};


