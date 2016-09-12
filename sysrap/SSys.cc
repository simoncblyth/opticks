#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cstring>


#include "SSys.hh"
#include "PLOG.hh"



void SSys::npdump(const char* path, const char* nptype)
{

    std::stringstream ss ; 
    ss << "python -c 'import os, numpy as np ; np.set_printoptions(suppress=True, precision=3) ; print np.load(os.path.expandvars(\""
       << path
       << "\")).view(" 
       << nptype 
       << ")' " 
    ;    

    std::string cmd = ss.str();
    LOG(info) << cmd ; 

    system(cmd.c_str());
}



int SSys::GetInteractivityLevel()
{
    // see opticks/notes/issues/automated_interop_tests.rst
    // hmm these envvars are potentially dependant on CMake/CTest version 
    // if that turns out to be the case will have to define an OPTICKS_INTERACTIVITY 
    // envvar for this purpose

    char* dtmd = getenv("DART_TEST_FROM_DART");           //  ctest run with --interactive-debug-mode 0
    char* cidm = getenv("CTEST_INTERACTIVE_DEBUG_MODE");  //  ctest run with --interactive-debug-mode 1  (the default)


    int level = 2 ;   
    if(dtmd && dtmd[0] == '1') 
    {
        level = 0 ;  // running under CTest with --interactive-debug-mode 0 meaning no-interactivity 
    }
    else if(IsRemoteSession())
    {
        level = 0 ;   // remote SSH running 
    }
    else if(cidm && cidm[0] == '1') 
    {
        level = 1 ;  // running under CTest with --interactive-debug-mode 1  
    }
    else
    {
        level = 2 ;  // not-running under CTest 
    }
    return level ;
}

bool SSys::IsRemoteSession()
{
    char* ssh_client = getenv("SSH_CLIENT");
    char* ssh_tty = getenv("SSH_TTY");

    bool is_remote = ssh_client != NULL || ssh_tty != NULL ; 

    LOG(trace) << "SSys::IsRemoteSession"
               << " ssh_client " << ssh_client 
               << " ssh_tty " << ssh_tty
               << " is_remote " << is_remote
               ; 

    return is_remote ; 
}


void SSys::WaitForInput(const char* msg)
{
    LOG(info) << "SSys::WaitForInput " << msg  ; 
    int c = '\0' ;
    do
    {
        c = std::cin.get() ;  

    } while(c != '\n' ); 
   
    LOG(info) << "SSys::WaitForInput DONE " ; 
}

int SSys::getenvint( const char* envkey, int fallback )
{
    char* val = getenv(envkey);
    int ival = val ? atoi_(val) : fallback ;
    return ival ; 
}

int SSys::atoi_( const char* a )
{
    std::string s(a);
    std::istringstream iss(s);
    int i ;
    iss >> i ; 
    return i ;
}


const char* SSys::getenvvar( const char* envvar )
{
    const char* evalue = getenv(envvar);
    LOG(debug) << "SSys::getenvvar"
              << " envvar " << envvar
              << " evalue " << evalue
              ;
    return evalue ; 
}


const char* SSys::getenvvar( const char* envprefix, const char* envkey, const char* fallback )
{
    char envvar[128];
    snprintf(envvar, 128, "%s%s", envprefix, envkey );
    const char* evalue = getenvvar(envvar) ; 
    return evalue ? evalue : fallback ; 
}

int SSys::setenvvar( const char* envprefix, const char* key, const char* value, bool overwrite)
{
    // heap as putenv does not copy


    std::stringstream ss ;
    if(envprefix) ss << envprefix ; 
    if(key)       ss << key ; 

    std::string ekey = ss.str();

    ss << "=" ;
    if(value) ss << value ; 

    std::string ekv = ss.str();

    const char* prior = getenv(ekey.c_str()) ;

    char* ekv_ = const_cast<char*>(strdup(ekv.c_str()));

    int rc = ( overwrite || !prior ) ? putenv(ekv_) : 0  ; 

    const char* after = getenv(ekey.c_str()) ;

    LOG(trace) << "SSys::setenvvar"
              << " ekey " << ekey 
              << " ekv " << ekv 
              << " overwrite " << overwrite
              << " prior " << ( prior ? prior : "NULL" )
              << " value " << ( value ? value : "NULL" )   
              << " after " << ( after ? after : "NULL" )   
              << " rc " << rc 
              ;

    return rc ;
} 




