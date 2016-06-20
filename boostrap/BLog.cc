#include <string>
#include <iostream>



#ifdef _MSC_VER
// signed/unsigned mismatch
#pragma warning( disable : 4018 )
#endif


#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include "boost/log/utility/setup.hpp"
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

// brap-
#include "BFile.hh"
#include "BLog.hh"
#include "BSys.hh"


#define DBG 1


// http://stackoverflow.com/questions/24302123/how-can-i-use-boost-log-across-dll-boundaries
// http://www.boost.org/doc/libs/1_61_0/libs/log/doc/html/log/installation/config.html

/*
If your application consists of more than one module (e.g. an exe and one or
several dll's) that use Boost.Log, the library must be built as a shared
object.
*/



BLog::BLog(int argc, char** argv)
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

void BLog::setPause(bool pause)
{
    m_pause = pause ; 
}
void BLog::setExitPause(bool pause)
{
    m_exitpause = pause ; 
}
void BLog::setName(const char* logname)
{
    if(logname) m_logname = strdup(logname) ;
}
void BLog::setLevel(const char* loglevel)
{
    if(loglevel) m_loglevel = strdup(loglevel) ;
}




void BLog::init()
{
    configure(m_argc, m_argv);
    setup(m_loglevel) ;

#ifdef DBG
    std::cerr << "BLog::init "
              << " loglevel " << ( m_loglevel ? m_loglevel : "NULL" )
              << std::endl ;    
#endif

    addFileLog();    // only if dir was provided 

#ifdef DBG
    std::cerr << "BLog::init"
               << " logname " << ( m_logname ? m_logname : "NULL" )
               << " loglevel " << m_loglevel 
               << " logdir " << ( m_logdir ? m_logdir : "NULL" )
              << std::endl ;    
              ;
#endif

    if(m_pause) BSys::WaitForInput("Blog::configure pausing...");
}


void BLog::configure(int argc, char** argv)
{
    const char* loglevel = "info" ; 
    const char* logdir = NULL ; 
    bool nogeocache = false ; 

    for(int i=1 ; i < argc ; ++i )
    {
        if(strcmp(argv[i], "-G")==0)              nogeocache = true ; 
        if(strcmp(argv[i], "--nogeocache")==0)    nogeocache = true ; 

        if(strcmp(argv[i], "--trace")==0)   loglevel = "trace" ;
        if(strcmp(argv[i], "--debug")==0)   loglevel = "debug" ;
        if(strcmp(argv[i], "--info")==0)    loglevel = "info" ;
        if(strcmp(argv[i], "--warning")==0) loglevel = "warning" ;
        if(strcmp(argv[i], "--error")==0)   loglevel = "error" ;
        if(strcmp(argv[i], "--fatal")==0)   loglevel = "fatal" ;

        if(i < argc - 1 && strcmp(argv[i], "--logdir")==0) logdir = argv[i+1] ;
                                    
        if(strcmp(argv[i], "--pause")==0)       setPause(true) ;    
        if(strcmp(argv[i], "--exitpause")==0)   setExitPause(true) ;    
    }

    if(argc > 0)
    {
        std::string stem = BFile::Stem(argv[0]);
        std::string logname(stem) ;
        if(nogeocache) logname += ".nogeocache" ; 
        logname += ".log" ;

#ifdef DBG
        std::cerr << "BLog::configure" 
                  << " logname " << logname 
                  << std::endl ; 
#endif

        setName(logname.c_str()); // logname derived from basename of executable, always available
    }

    setLevel(loglevel); // level always defaulted 
    setDir(logdir);     // dir often not set yet 

#ifdef DBG
   std::cerr << "BLog::configure" 
             << " logname " << ( m_logname ? m_logname : "NULL" )
             << " loglevel " << m_loglevel
             << std::endl  
              ;  
#endif

}


int BLog::FilterLevel(int argc, char** argv )
{
    int ll = boost::log::trivial::info ;
    for(int i=1 ; i < argc ; ++i )
    {
        if(strcmp(argv[i], "--trace")==0)   
            ll = boost::log::trivial::trace ; 

        if(strcmp(argv[i], "--debug")==0)   
            ll = boost::log::trivial::debug ; 

        if(strcmp(argv[i], "--info")==0)    
            ll = boost::log::trivial::info ; 

        if(strcmp(argv[i], "--warning")==0) 
            ll = boost::log::trivial::warning ; 

        if(strcmp(argv[i], "--error")==0)   
            ll = boost::log::trivial::error ; 

        if(strcmp(argv[i], "--fatal")==0)   
            ll = boost::log::trivial::fatal ; 
    }
    return ll ;
}

int BLog::FilterLevel(const char* level_)
{
    int ll = boost::log::trivial::info ;
    {
        std::string level(level_);
        if(level.compare("trace") == 0) 
            ll = boost::log::trivial::trace ; 

        if(level.compare("debug") == 0) 
            ll = boost::log::trivial::debug ; 

        if(level.compare("info") == 0)  
            ll = boost::log::trivial::info ; 

        if(level.compare("warning") == 0)  
            ll = boost::log::trivial::warning ; 

        if(level.compare("error") == 0)  
            ll = boost::log::trivial::error ; 

        if(level.compare("fatal") == 0)  
            ll = boost::log::trivial::fatal ; 
    }
    return ll ;
}


void BLog::setFilter(const char* level_)
{
   // doesnt work across dll divide ? no error just no filtering

    boost::log::core::get()->set_filter
    (    
        boost::log::trivial::severity >=  FilterLevel(level_)
    ); 

#ifdef DBG
     std::cerr
               << "BLog::setFilter"
               << " level_ " << level_ 
               << std::endl 
               ; 
#endif
}



void BLog::setup(const char* level_)
{
    m_setup = true ; 

    setFilter(level_);


/*
    boost::log::add_common_attributes();
    boost::log::register_simple_formatter_factory< boost::log::trivial::severity_level, char >("Severity");  

    boost::log::add_console_log(
        std::cerr, 
        boost::log::keywords::format = "[%TimeStamp%]:%Severity%: %Message%",
        boost::log::keywords::auto_flush = true 
    );   

*/



}


void BLog::setDir(const char* logdir)
{
    if(m_setup) 
    {
         std::cerr
               << "BLog::setDir"
               << " logdir " << ( logdir ? logdir : "NULL" )
               << std::endl 
                   ;
    }

    if(logdir) m_logdir = strdup(logdir) ;
    addFileLog();
}


void BLog::addFileLog()
{
    if(m_setup) 
    {
        std::cerr
               << "BLog::addFileLog"
               << " logname " << ( m_logname ? m_logname : "NULL" )
               << " logdir " << ( m_logdir ? m_logdir : "NULL" )
               << " addfile " << m_addfile 
               << " setup " << m_setup
               << std::endl 
               ;
    }

    if(m_logname && m_logdir && !m_addfile && m_setup)
    {
       m_addfile = true ; 

       BFile::CreateDir(m_logdir);
       std::string logpath = BFile::FormPath(m_logdir, m_logname) ;
       boost::log::add_file_log(
              logpath,
              boost::log::keywords::format = "[%TimeStamp%]:%Severity%: %Message%"
             );

#ifdef DBG
       std::cerr  << "BLog::addFileLog"
                  << " logpath " << logpath 
                  << std::endl 
                  ;

#endif
    }
}


BLog::~BLog()
{
    if(m_exitpause) BSys::WaitForInput("Blog::~BLog exit-pausing...");
}



