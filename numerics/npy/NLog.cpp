#include "NLog.hpp"

#include <string>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


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

    fs::path logdir(ldir);
    if(!fs::exists(logdir))
    {    
        if (fs::create_directories(logdir))
        {    
            printf("logging_init: created directory %s \n", ldir) ;
        }    
    }    

    fs::path logpath(logdir / lname );

    const char* path = logpath.string().c_str(); 

    boost::log::add_file_log(path);

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


    LOG(info) << "logging_init " << path ; 
}



void NLog::configure(int argc, char** argv)
{
    // full argument parsing is done in App::config, 
    // but logging setup needs to happen before that 

    const char* logname = NULL ; 
    const char* loglevel = NULL ; 

    for(unsigned int i=1 ; i < argc ; ++i )
    {
       // TODO: find a more general way to switch logname based on a particular arg 

        if(strcmp(argv[i], "-G")==0)        logname = "ggeoview.nogeocache.log" ;

        if(strcmp(argv[i], "--trace")==0)   loglevel = "trace" ;
        if(strcmp(argv[i], "--debug")==0)   loglevel = "debug" ;
        if(strcmp(argv[i], "--info")==0)    loglevel = "info" ;
        if(strcmp(argv[i], "--warning")==0) loglevel = "warning" ;
        if(strcmp(argv[i], "--error")==0)   loglevel = "error" ;
        if(strcmp(argv[i], "--fatal")==0)   loglevel = "fatal" ;
    }

    // dont print anything here, it messes with --idp
    //printf(" logname: %s loglevel: %s\n", logname, loglevel );

    if(logname)
        m_logname = strdup(logname);

    if(loglevel)
        m_loglevel = strdup(loglevel);

}


void NLog::init(const char* idpath)
{
    logging_init(idpath, m_logname, m_loglevel );
}






