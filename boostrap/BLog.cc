#include "BLog.hh"
#include "BSys.hh"

#include <string>
#include <iostream>

// bregex-
#include "dbg.hh"
#include "fsutil.hh"

#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include "boost/log/utility/setup.hpp"
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void logging_init(const char* ldir, const char* lname, const char* level_)
{
   // see blogg-

    unsigned int ll = boost::log::trivial::info ;
 
    std::string level(level_);
    if(level.compare("trace") == 0) ll = boost::log::trivial::trace ; 
    if(level.compare("debug") == 0) ll = boost::log::trivial::debug ; 
    if(level.compare("info") == 0)  ll = boost::log::trivial::info ; 
    if(level.compare("warning") == 0)  ll = boost::log::trivial::warning ; 
    if(level.compare("error") == 0)  ll = boost::log::trivial::error ; 
    if(level.compare("fatal") == 0)  ll = boost::log::trivial::fatal ; 

    DBG("BLog::","lname ", lname)   ; 
    DBG("BLog::","ldir", ldir )   ; 

    fsutil::CreateDir(ldir);

    DBG("BLog::","ldir created", ldir )   ; 


    std::string logpath = fsutil::FormPath(ldir, lname) ;
    const char* path = logpath.c_str(); 


    DBG("BLog::","logpath string", logpath )  ; 
    DBG("BLog::","logpath c_str", path)  ; 

    boost::log::add_file_log(logpath);

    DBG("BLog::","logpath added", path)  ; 

    boost::log::core::get()->set_filter
    (    
        boost::log::trivial::severity >= ll
    );   

    boost::log::add_common_attributes();

    boost::log::register_simple_formatter_factory< boost::log::trivial::severity_level, char >("Severity");  

    boost::log::add_console_log(
        std::cerr, 
        boost::log::keywords::format = "[%TimeStamp%]:%Severity%: %Message%",
        boost::log::keywords::auto_flush = true 
    );   


    LOG(debug) << "BLog logging_init " << path ; 
}



void BLog::configure(int argc, char** argv, const char* idpath)
{
    // full argument parsing is done in App::config, 
    // but logging setup needs to happen before that 

    const char* logname = NULL ; 
    const char* loglevel = NULL ; 

    for(int i=1 ; i < argc ; ++i )
    {
       // TODO: find a more general way to switch logname based on a particular arg 

        if(strcmp(argv[i], "-G")==0)        logname = "ggeoview.nogeocache.log" ;

        if(strcmp(argv[i], "--trace")==0)   loglevel = "trace" ;
        if(strcmp(argv[i], "--debug")==0)   loglevel = "debug" ;
        if(strcmp(argv[i], "--info")==0)    loglevel = "info" ;
        if(strcmp(argv[i], "--warning")==0) loglevel = "warning" ;
        if(strcmp(argv[i], "--error")==0)   loglevel = "error" ;
        if(strcmp(argv[i], "--fatal")==0)   loglevel = "fatal" ;

                                    
        if(strcmp(argv[i], "--pause")==0)   setPause(true) ;    
    }

    // dont print anything here, it messes with --idp
    //printf(" logname: %s loglevel: %s\n", logname, loglevel );

    if(logname)
        m_logname = strdup(logname);

    if(loglevel)
        m_loglevel = strdup(loglevel);

    if(idpath)
        init(idpath) ;


    std::cout << "BLog::configure" << " logname " << m_logname << " loglevel " << m_loglevel << std::endl ;  


    if(m_pause) BSys::WaitForInput("Blog::configure pausing...");

}


void BLog::init(const char* idpath)
{

    std::cout << "BLog::init" << " logname " << m_logname << " loglevel " << m_loglevel << " idpath " << ( idpath ? idpath : "NULL" ) << std::endl ;  


    logging_init(idpath, m_logname, m_loglevel );
}





