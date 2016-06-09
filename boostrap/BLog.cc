#include <string>
#include <iostream>

#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include "boost/log/utility/setup.hpp"
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

// brap-
#include "fsutil.hh"
#include "BLog.hh"
#include "BSys.hh"


void BLog::init()
{
    configure(m_argc, m_argv);
    setup(m_loglevel) ;

    addFileLog();    // only if dir was provided 

    LOG(trace) << "BLog::init"
               << " logname " << m_logname
               << " loglevel " << m_loglevel 
               << " logdir " << ( m_logdir ? m_logdir : "NULL" )
              ;

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
        std::string stem = fsutil::Stem(argv[0]);
        std::string logname(stem) ;
        if(nogeocache) logname += ".nogeocache" ; 
        logname += ".log" ;
        // LOG(trace) << "BLog::configure" << " logname " << logname ;
        setName(logname.c_str()); // logname derived from basename of executable, always available
    }

    setLevel(loglevel); // level always defaulted 
    setDir(logdir);     // dir often not set yet 

   // LOG(trace) << "BLog::configure" << " logname " << m_logname << " loglevel " << m_loglevel ;  

}



void BLog::setup(const char* level_)
{
    m_setup = true ; 

    boost::log::add_common_attributes();

    boost::log::register_simple_formatter_factory< boost::log::trivial::severity_level, char >("Severity");  

    boost::log::add_console_log(
        std::cerr, 
        boost::log::keywords::format = "[%TimeStamp%]:%Severity%: %Message%",
        boost::log::keywords::auto_flush = true 
    );   

    unsigned int ll = boost::log::trivial::info ;
 
    std::string level(level_);

    if(level.compare("trace") == 0) ll = boost::log::trivial::trace ; 
    if(level.compare("debug") == 0) ll = boost::log::trivial::debug ; 
    if(level.compare("info") == 0)  ll = boost::log::trivial::info ; 
    if(level.compare("warning") == 0)  ll = boost::log::trivial::warning ; 
    if(level.compare("error") == 0)  ll = boost::log::trivial::error ; 
    if(level.compare("fatal") == 0)  ll = boost::log::trivial::fatal ; 

    boost::log::core::get()->set_filter
    (    
        boost::log::trivial::severity >= ll
    ); 

    LOG(trace) << "BLog::setup"
               << " level " << level 
               << " ll " << ll
               ; 
}


void BLog::setDir(const char* logdir)
{
    if(m_setup) 
        LOG(trace) << "BLog::setDir"
                   << " logdir " << ( logdir ? logdir : "NULL" )
                   ;

    if(logdir) m_logdir = strdup(logdir) ;
    addFileLog();
}


void BLog::addFileLog()
{
    if(m_setup) 
    LOG(trace) << "BLog::addFileLog"
               << " logname " << ( m_logname ? m_logname : "NULL" )
               << " logdir " << ( m_logdir ? m_logdir : "NULL" )
               << " addfile " << m_addfile 
               << " setup " << m_setup
               ;

    if(m_logname && m_logdir && !m_addfile && m_setup)
    {
       m_addfile = true ; 

       fsutil::CreateDir(m_logdir);
       std::string logpath = fsutil::FormPath(m_logdir, m_logname) ;
       boost::log::add_file_log(
              logpath,
              boost::log::keywords::format = "[%TimeStamp%]:%Severity%: %Message%"
             );

       LOG(info) << "BLog::addFileLog"
                  << " logpath " << logpath 
                  ;
    }
}


BLog::~BLog()
{
    if(m_exitpause) BSys::WaitForInput("Blog::~BLog exit-pausing...");
}



