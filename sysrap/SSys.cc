#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>

#include <cassert>
#include <cstdio>
#include <fstream> 

#include <sys/wait.h>

#include "OKConf.hh"

#include "SSys.hh"
#include "PLOG.hh"


/*
simon:optixrap blyth$ python -c 'import sys, os, numpy as np ; sys.exit(214) '
simon:optixrap blyth$ echo $?
214
*/



const unsigned SSys::SIGNBIT32  = 0x80000000 ;
const unsigned SSys::OTHERBIT32 = 0x7fffffff ;


const char* SSys::fmt(const char* tmpl, unsigned val)
{
    char buf[100] ; 
    snprintf(buf, 100, tmpl, val );
    return strdup(buf);
}


int SSys::exec(const char* exe, const char* path)
{
    std::stringstream ss ; 
    ss << exe << " " << path ;
    std::string cmd = ss.str();
    return SSys::run(cmd.c_str());
}

int SSys::run(const char* cmd)
{
    int rc_raw = system(cmd);
    int rc =  WEXITSTATUS(rc_raw) ;

    LOG(info) << cmd 
              << " rc_raw : " << rc_raw 
              << " rc : " << rc
              ;

    if(rc != 0)
    {
        LOG(warning) << "SSys::run FAILED with "
                     << " cmd " << cmd 
                     ;
        LOG(trace) << " possibly you need to set export PATH=$OPTICKS_HOME/ana:$OPTICKS_HOME/bin:/usr/local/opticks/lib:$PATH " ;
    }
    


    return rc ;  
}

int SSys::npdump(const char* path, const char* nptype, const char* postview, const char* printoptions)
{
    if(!printoptions) printoptions = "suppress=True,precision=3" ;

    std::stringstream ss ; 
    ss << "python -c 'import sys, os, numpy as np ;"
       << " np.set_printoptions(" << printoptions << ") ;"
       << " a=np.load(os.path.expandvars(\"" << path << "\")) ;"
       << " print a.shape ;"
       << " print a.view(" << nptype << ")" << ( postview ? postview : "" ) << " ;"
       << " sys.exit(0) ' " 
    ;    

    std::string cmd = ss.str();
    return run(cmd.c_str());
}


void SSys::xxdump(char* buf, int num_bytes, int width, char non_printable )
{
     LOG(info) << " SSys::xxdump "
             << " '0' " << (int)'0' 
             << " '9' " << (int)'9' 
             << " 'A' " << (int)'A' 
             << " 'Z' " << (int)'Z'
             << " 'a' " << (int)'a'
             << " 'z' " << (int)'z'
             ;

    for(unsigned i=0 ; i < num_bytes ; i++) 
    {   
        char c = buf[i] ; 
        bool printable = c >= ' ' && c <= '~' ;  // https://en.wikipedia.org/wiki/ASCII
        std::cout << ( printable ? c : non_printable )  ;
        if((i+1) % width == 0 ) std::cout << "\n" ; 
   }   
}








int SSys::OKConfCheck()
{
    int rc = OKConf::Check(); 
    OKConf::Dump();
    assert( rc == 0 );
    return rc ; 
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

bool SSys::IsENVVAR(const char* envvar)
{
    char* e = getenv(envvar);
    return e != NULL ;
}

bool SSys::IsVERBOSE() {  return IsENVVAR("VERBOSE") ; }
bool SSys::IsHARIKARI() { return IsENVVAR("HARIKARI") ; }



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


int SSys::setenvvar( const char* ekey, const char* value, bool overwrite)
{
    std::stringstream ss ;
    ss << ekey << "=" ;
    if(value) ss << value ; 

    std::string ekv = ss.str();

    const char* prior = getenv(ekey) ;

    char* ekv_ = const_cast<char*>(strdup(ekv.c_str()));

    int rc = ( overwrite || !prior ) ? putenv(ekv_) : 0  ; 

    const char* after = getenv(ekey) ;

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



int SSys::setenvvar( const char* envprefix, const char* key, const char* value, bool overwrite)
{
    std::stringstream ss ;
    if(envprefix) ss << envprefix ; 
    if(key)       ss << key ; 
    std::string ekey = ss.str();
    return SSys::setenvvar(ekey.c_str(), value, overwrite );    
} 




unsigned SSys::COUNT = 0 ; 

void SSys::Dump_(const char* msg)
{
    std::cout << std::setw(3) << COUNT << "[" << std::setw(20) << "std::cout" << "] " << msg << std::endl;  
    std::cerr << std::setw(3) << COUNT << "[" << std::setw(20) << "std::cerr" << "] " << msg << std::endl;  
    printf("%3d[%20s] %s \n", COUNT, "printf", msg );  
    std::printf("%3d[%20s] %s \n", COUNT, "std::printf", msg );  
}
void SSys::Dump(const char* msg)
{
    Dump_(msg);
    std::cerr << std::endl  ;   
    COUNT++ ; 
}





