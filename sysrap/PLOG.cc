#include <cstring>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <string>

#include "PLOG.hh"

PLOG* PLOG::instance = NULL ; 

#define MAXARGC 50 

//#define PLOG_DBG 1


void PLOG::_dump(const char* msg, int argc, char** argv)
{
    std::cerr <<  msg
              << " argc " << argc ;

    for(int i=0 ; i < argc ; i++) std::cerr << argv[i] ; 
    std::cerr << std::endl ;               
}

int PLOG::_parse(int argc, char** argv, const char* fallback)
{
    // Parse arguments case insensitively looking for --VERBOSE --info --error etc.. returning global logging level

    assert( argc < MAXARGC && " argc sanity check fail "); 

    std::string ll = fallback ; 
    for(int i=1 ; i < argc ; ++i )
    {
        std::string arg(argv[i] ? argv[i] : "");
        std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);
        if(arg.compare("--trace")==0)   ll = "VERBOSE" ; 
        if(arg.compare("--verbose")==0) ll = "VERBOSE" ; 
        if(arg.compare("--debug")==0)   ll = "DEBUG" ; 
        if(arg.compare("--info")==0)    ll = "INFO" ; 
        if(arg.compare("--warning")==0) ll = "WARNING" ; 
        if(arg.compare("--error")==0)   ll = "ERROR" ; 
        if(arg.compare("--fatal")==0)   ll = "FATAL" ;

        // severityFromString uses first char 
    }
     
    std::transform(ll.begin(), ll.end(), ll.begin(), ::toupper);
    plog::Severity severity = plog::severityFromString(ll.c_str()) ;

    int level = static_cast<int>(severity); 

    //_dump("PLOG::parse", argc, argv );

    return level ;  
}



const char* PLOG::_logpath_parse(int argc, char** argv)
{
    assert( argc < MAXARGC && " argc sanity check fail "); 
    //  Construct logfile path based on executable name argv[0] with .log appended 
    std::string lp(argc > 0 ? argv[0] : "default") ; 
    lp += ".log" ; 
    return strdup(lp.c_str());
}


int PLOG::_prefixlevel_parse(int argc, char** argv, const char* fallback, const char* prefix)
{
    // Parse commandline to find project logging level  
    // looking for a single project prefix, eg 
    // with the below commandline and prefix of sysrap
    // the level "error" should be set.
    //
    // When no level is found the fallback level is used.
    //
    //    --okcore info --sysrap error --brap trace --npy trace
    //  
    // Both prefix and the arguments are lowercased before comparison.
    //

    assert( argc < MAXARGC && " argc sanity check fail "); 

    std::string pfx(prefix);
    std::transform(pfx.begin(), pfx.end(), pfx.begin(), ::tolower);
    std::string apfx("--");
    apfx += pfx ;  

    std::string ll(fallback) ;
    for(int i=1 ; i < argc ; ++i )
    {
        char* ai = argv[i] ;
        char* aj = i + 1 < argc ? argv[i+1] : NULL ; 

        std::string arg(ai ? ai : "");
        std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);
        //std::cerr << arg << std::endl ; 

        if(arg.compare(apfx) == 0 && aj != NULL ) ll.assign(aj) ;
    }

    std::transform(ll.begin(), ll.end(), ll.begin(), ::toupper);

    const char* llc = ll.c_str();
    plog::Severity severity = strcmp(llc, "TRACE")==0 ? plog::severityFromString("VERB") : plog::severityFromString(llc) ;
    int level = static_cast<int>(severity); 

    //_dump("PLOG::prefix_parse", argc, argv );

    return level ; 
}




int PLOG::parse(plog::Severity _fallback)
{
    const char* fallback = _name(_fallback);
    return parse(fallback);
}
int PLOG::parse(const char* fallback)
{
    int ll = _parse(args._argc, args._argv, fallback);

#ifdef PLOG_DBG
    std::cerr << "PLOG::parse"
              << " fallback " << fallback
              << " level " << ll 
              << " name " << _name(ll)
              << std::endl ;
#endif

    return ll ; 
}


int PLOG::prefixlevel_parse(int _fallback, const char* prefix)
{
    plog::Severity fallback = static_cast<plog::Severity>(_fallback); 
    return prefixlevel_parse(fallback, prefix) ; 
}

int PLOG::prefixlevel_parse(plog::Severity _fallback, const char* prefix)
{
    const char* fallback = _name(_fallback);
    return prefixlevel_parse(fallback, prefix) ; 
}
int PLOG::prefixlevel_parse(const char* fallback, const char* prefix)
{
    int ll =  _prefixlevel_parse(args._argc, args._argv, fallback, prefix);

#ifdef PLOG_DBG
    std::cerr << "PLOG::prefixlevel_parse"
              << " fallback " << fallback
              << " prefix " << prefix 
              << " level " << ll 
              << " name " << _name(ll)
              << std::endl ;
#endif

    return ll ; 
}



const char* PLOG::_name(int level)
{
   plog::Severity severity  = static_cast<plog::Severity>(level); 
   return plog::severityToString(severity);
}
const char* PLOG::_name(plog::Severity severity)
{
   return plog::severityToString(severity);
}

const char* PLOG::name()
{
   plog::Severity severity  = static_cast<plog::Severity>(level); 
   return _name(severity);
}


PLOG::PLOG(int argc_, char** argv_, const char* fallback, const char* prefix)
    :
      args(argc_, argv_, "OPTICKS_ARGS" , ' '),   // when argc_ is 0 the named envvar is checked for arguments instead 
      level(info),
      logpath(_logpath_parse(argc_, argv_)),
      logmax(3)
{
   level = prefix == NULL ?  parse(fallback) : prefixlevel_parse(fallback, prefix ) ;    

   assert( instance == NULL && "ONLY EXPECTING A SINGLE PLOG INSTANCE" );
   instance = this ; 

/*
   std::cerr << "PLOG::PLOG " 
             << " instance " << instance 
             << " this " << this 
             << " logpath " << logpath
             << std::endl
             ;
*/

}



