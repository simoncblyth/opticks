/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <cmath>
#include <math.h>

#include <cassert>
#include <cstdio>
#include <fstream> 

#include <sys/wait.h>

#include "OKConf.hh"

#include "SSys.hh"
#include "SStr.hh"
#include "PLOG.hh"


/*
simon:optixrap blyth$ python -c 'import sys, os, numpy as np ; sys.exit(214) '
simon:optixrap blyth$ echo $?
214
*/

extern char **environ;



const plog::Severity SSys::LEVEL = PLOG::EnvLevel("SSys", "DEBUG")  ; 


void SSys::DumpEnv(const char* pfx ) // static
{
    int i = 1;
    for (char* s = *environ ; s; i++) 
    {
       if(pfx == NULL || strncmp(s, pfx, strlen(pfx)) == 0) printf("%s\n", s);
       s = *(environ+i);
    }
}


const unsigned SSys::SIGNBIT32  = 0x80000000 ;
const unsigned SSys::OTHERBIT32 = 0x7fffffff ;


bool SSys::IsNegativeZero(float f)
{
   return std::signbit(f) && f == -0.f ;  
}


const char* SSys::fmt(const char* tmpl, unsigned val)
{
    char buf[100] ; 
    snprintf(buf, 100, tmpl, val );
    return strdup(buf);
}


int SSys::exec(const char* exe, const char* arg1, const char* arg2)
{
    std::stringstream ss ; 
    ss << exe << " " ; 
    if(arg1) ss << arg1 << " " ; 
    if(arg2) ss << arg2 << " " ; 

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
        LOG(error) 
            << "FAILED with "
            << " cmd " << cmd 
            << " RC " << rc 
            ;
        LOG(verbose) << " possibly you need to set export PATH=$OPTICKS_HOME/ana:$OPTICKS_HOME/bin:/usr/local/opticks/lib:$PATH " ;
    }
    return rc ;  
}









std::string SSys::Which(const char* script)
{
    bool chomp = true ; 

    int rc(0); 
    std::string path = SSys::POpen("which 2>/dev/null", script, chomp, rc );
    LOG(LEVEL) 
         << " script " << script
         << " path " << path 
         << " rc " << rc
         ;

    std::string empty ; 
    return rc == 0 ? path : empty ; 
}


/**
SSys::POpen
-------------
Run command and get output into string, 
Newlines are removed when chomp is true.

**/


std::string SSys::POpen(const char* cmd, bool chomp, int& rc)
{
    LOG(LEVEL) << "[ " << cmd ; 

    std::stringstream ss ; 
    FILE *fp = popen(cmd, "r");
    char line[512];    
    while (fgets(line, sizeof(line), fp) != NULL) 
    {
       if(chomp) line[strcspn(line, "\n")] = 0;
       //LOG(info) << "[" << line << "]" ; 
       ss << line ;  
    }

    rc=0 ; 
    int st = pclose(fp);
    if(WIFEXITED(st)) rc=WEXITSTATUS(st);
    LOG(LEVEL) << "] " << cmd << " rc " << rc ; 

    return ss.str() ; 
}

std::string SSys::POpen(const char* cmda, const char* cmdb, bool chomp, int& rc)
{
    std::stringstream ss ; 
    if(cmda) ss << cmda ; 
    ss << " " ; 
    if(cmdb) ss << cmdb ; 

    std::string s = ss.str(); 
    return POpen(s.c_str(), chomp, rc ); 
}









int SSys::npdump(const char* path, const char* nptype, const char* postview, const char* printoptions)
{
    if(!printoptions) printoptions = "suppress=True,precision=3" ;

    std::stringstream ss ; 
    ss << "python -c 'import sys, os, numpy as np ;"
       << " np.set_printoptions(" << printoptions << ") ;"
       << " a=np.load(os.path.expandvars(\"" << path << "\")) ;"
       << " print(a.shape);"
       << " print(a.view(" << nptype << ")" << ( postview ? postview : "" ) << ") ;"
       << " sys.exit(0) ' " 
    ;    

    std::string cmd = ss.str();
    return run(cmd.c_str());
}


void SSys::xxdump(char* buf, int num_bytes, int width, char non_printable )
{

/*
     LOG(info) << " SSys::xxdump "
             << " '0' " << (int)'0' 
             << " '9' " << (int)'9' 
             << " 'A' " << (int)'A' 
             << " 'Z' " << (int)'Z'
             << " 'a' " << (int)'a'
             << " 'z' " << (int)'z'
             ;
*/

    for(int i=0 ; i < num_bytes ; i++) 
    {   
        char c = buf[i] ; 
        bool printable = c >= ' ' && c <= '~' ;  // https://en.wikipedia.org/wiki/ASCII
        std::cout << ( printable ? c : non_printable )  ;
        if((i+1) % width == 0 ) std::cout << "\n" ; 
   }   
}


std::string SSys::xxd( char* buf, int num_bytes, int width, char non_printable )
{
    std::stringstream ss ;  
    for(int i=0 ; i < num_bytes ; i++) 
    {   
        char c = buf[i] ; 
        bool printable = c >= ' ' && c <= '~' ;  // https://en.wikipedia.org/wiki/ASCII
        ss << ( printable ? c : non_printable )  ;
        if((i+1) % width == 0 ) ss << "\n" ; 
   }   
   return ss.str(); 
}



std::string SSys::hexlify(const void* obj, size_t size, bool reverse)
{ 
    const unsigned char * const bytes = static_cast<const unsigned char *>(obj);
    std::stringstream ss ; 
    for(size_t i=0 ; i < size ; i++) ss << std::setw(2) << std::hex << std::setfill('0') << unsigned(bytes[reverse ? size - 1 - i : i]) ; 
    return ss.str(); 
}





int SSys::OKConfCheck()
{
    int rc = OKConf::Check(); 
    OKConf::Dump();
    assert( rc == 0 );
    return rc ; 
}


/**
SSys::GetInteractivityLevel
-------------------------------

Defines an interactivity level of the session 
as discerned from the existance of certain environment
variables characteristic of test running and remote ssh sessions.

see opticks/notes/issues/automated_interop_tests.rst

hmm these envvars are potentially dependant on CMake/CTest version 
if that turns out to be the case will have to define an OPTICKS_INTERACTIVITY 
envvar for this purpose


This is used by OpticksMode::getInteractivityLevel see `opticks-f getInteractivityLevel`



**/

int SSys::GetInteractivityLevel()
{
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
    else if(cidm && cidm[0] == '0') 
    {
        level = 0 ;  
    } 
    else
    {
        level = 2 ;  // not-running under CTest 
    }
    return level ;
}



bool SSys::IsCTestRunning()
{
    char* dtmd = getenv("DART_TEST_FROM_DART");           //  ctest run with --interactive-debug-mode 0
    char* cidm = getenv("CTEST_INTERACTIVE_DEBUG_MODE");  //  ctest run with --interactive-debug-mode 1  (the default)
    return dtmd || cidm ; 
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

    LOG(verbose) << "SSys::IsRemoteSession"
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

char SSys::getenvchar( const char* envkey, char fallback )
{
    char* val = getenv(envkey);
    char c = val ? val[0] : fallback ;
    return c ; 
}

unsigned SSys::getenvunsigned( const char* envkey, unsigned fallback )
{
    char* val = getenv(envkey);
    unsigned uval = val ? atoi_(val) : fallback ;
    return uval ; 
}

float SSys::getenvfloat( const char* envkey, float fallback )
{
    char* val = getenv(envkey);
    float fval = val ? atof_(val) : fallback ;
    return fval ; 
}

bool SSys::getenvbool( const char* envkey )
{
    char* val = getenv(envkey);
    bool ival = val ? true : false ;
    return ival ; 
}




int SSys::getenvintvec( const char* envkey, std::vector<int>& ivec, char delim, const char* fallback  )
{
    char* line = getenv(envkey);
    std::stringstream ss; 
    ss.str(line ? line : fallback)  ;

    std::string s;
    while (std::getline(ss, s, delim)) ivec.push_back(atoi_(s.c_str())) ; 
    
    return ivec.size(); 
}




int SSys::atoi_( const char* a )
{
    std::string s(a);
    std::istringstream iss(s);
    int i ;
    iss >> i ; 
    return i ;
}


float SSys::atof_( const char* a )
{
    std::string s(a);
    std::istringstream iss(s);
    float f ;
    iss >> f ; 
    return f ;
}

void SSys::split(std::vector<std::string>& elem, const char* str, char delim )
{
    if(strchr(str, delim) == nullptr )
    {
        elem.push_back(str); 
    } 
    else
    {
        std::stringstream ss; 
        ss.str(str)  ;
        std::string s;
        while (std::getline(ss, s, delim)) elem.push_back(s) ; 
    } 
}

/**
SSys::getenvvar
------------------

When the argument contains ',' it is split with this delimiter
into separate keys which are checked in order.
The first with non-null value is returned.

**/

const char* SSys::getenvvar( const char* envvar )
{
    std::vector<std::string> evars ; 
    split(evars, envvar, ','); 
    const char* evalue = nullptr ; 
    for(unsigned i=0 ; i < evars.size() ; i++)
    {
        const char* ekey = evars[i].c_str(); 
        evalue = getenv(ekey);
        if(evalue) break ; 
    }
    return evalue ; 
}

const char* SSys::getenvvar( const char* envkey, const char* fallback )
{
    const char* evalue = getenvvar(envkey) ; 
    return evalue ? evalue : fallback ; 
}

const char* SSys::username()
{
#ifdef _MSC_VER
    const char* user = SSys::getenvvar("USERNAME") ;
#else
    const char* user = SSys::getenvvar("USER") ;
#endif
    return user ? user : "SSys-username-undefined" ; 
}


#ifdef _MSC_VER
const char* SSys::hostname()
{
    // https://stackoverflow.com/questions/27914311/get-computer-name-and-logged-user-name
    return NULL ; 
}
#else

#ifndef HOST_NAME_MAX
# if defined(_POSIX_HOST_NAME_MAX)
#  define HOST_NAME_MAX _POSIX_HOST_NAME_MAX
# elif defined(MAXHOSTNAMELEN)
#  define HOST_NAME_MAX MAXHOSTNAMELEN
# endif
#endif /* HOST_NAME_MAX */


#include <unistd.h>
#include <limits.h>
const char* SSys::hostname()
{
    char hostname[HOST_NAME_MAX];
    hostname[0] = '\0' ;
    gethostname(hostname, HOST_NAME_MAX);
    return hostname[0] == '\0' ? "SSys-hostname-undefined" : strdup(hostname) ; 
}
#endif


/**
SSys::setenvvar
-----------------

As shell handling of empty strings is inconvenient the special_empty_token char 
allows a single char to represent the empty string, eg '-' 

**/

int SSys::setenvvar( const char* ekey, const char* value, bool overwrite, char special_empty_token)
{
    std::stringstream ss ;
    ss << ekey << "=" ;

    if(value)
    {
        if(special_empty_token != '\0' && strlen(value) == 1 && value[0] == special_empty_token)
        {
            ss << "" ; 
        }
        else
        {
            ss << value ; 
        }
    }

    std::string ekv = ss.str();
    const char* prior = getenv(ekey) ;

    char* ekv_ = const_cast<char*>(strdup(ekv.c_str()));

    int rc = ( overwrite || !prior ) ? putenv(ekv_) : 0  ; 

    const char* after = getenv(ekey) ;

    LOG(debug) 
        << " ekey " << ekey 
        << " ekv " << ekv 
        << " overwrite " << overwrite
        << " prior " << ( prior ? prior : "NULL" )
        << " value " << ( value ? value : "NULL" )   
        << " after " << ( after ? after : "NULL" )   
        << " rc " << rc 
        ;

    //std::raise(SIGINT);  
    return rc ;
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



const char* SSys::ResolveExecutable(const char* envvar_key, const char* default_executable)
{
    const char* ev = SSys::getenvvar(envvar_key, default_executable) ; 
    if(ev && ev[0] == '/') return strdup(ev) ;  // assume absolute path of python executable is provided
    std::string path = SSys::Which(ev); 
    return strdup(path.c_str()); 
}
const char* SSys::ResolvePython()
{
    return SSys::ResolveExecutable("OPTICKS_PYTHON", "python"); 
}

/**
SSys::RunPythonScript
-----------------------

Control which python to use with envvar OPTICKS_PYTHON eg::

    OPTICKS_PYTHON=python2.7 SSysTest 
    OPTICKS_PYTHON=python3   SSysTest
    OPTICKS_PYTHON=/Users/blyth/miniconda3/bin/python3 SSysTest 

**/

int SSys::RunPythonScript(const char* script)
{
    std::string script_path = SSys::Which(script);
    const char* python_executable = SSys::ResolvePython() ; 
    LOG(info) 
         << " script " << script
         << " script_path " << script_path 
         << " python_executable " << python_executable 
         ;   

    int RC = script_path.empty() ? 101 : SSys::exec(python_executable,script_path.c_str()) ;
    LOG(info) << " RC " << RC ; 
    return RC ; 
}

int SSys::RunPythonCode(const char* code)
{
    const char* python_executable = SSys::ResolvePython() ; 
    LOG(info) 
         << " code [" << code  << "]"
         << " python_executable " << python_executable 
         ;   

    const char* arg1 = "-c" ; 
    const char* arg2 = SStr::Concat("'", code, "'") ;  

    int RC = code == NULL ? 101 : SSys::exec(python_executable,arg1, arg2) ;
    LOG(info) << " RC " << RC ; 
    return RC ; 
}


