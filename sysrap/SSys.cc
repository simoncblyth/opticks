#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cstring>


#include "SSys.hh"
#include "PLOG.hh"


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


const char* SSys::getenvvar( const char* envprefix, const char* envkey, const char* fallback )
{
    char envvar[128];
    snprintf(envvar, 128, "%s%s", envprefix, envkey );
    const char* evalue = getenv(envvar);
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




