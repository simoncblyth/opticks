#include <iostream>
#include <cstdlib>
#include <cstring>

#include <boost/lexical_cast.hpp>

#include "BSys.hh"
#include "BLog.hh"


void BSys::WaitForInput(const char* msg)
{
    LOG(info) << "BSys::WaitForInput " << msg  ; 
    char c = '\0' ;
    do
    {
        c = std::cin.get() ;  

    } while(c != '\n' ); 
   
    LOG(info) << "BSys::WaitForInput DONE " ; 
}

int BSys::getenvint( const char* envkey, int fallback )
{
    char* val = getenv(envkey);
    int ival = val ? boost::lexical_cast<int>(val) : fallback ;
    return ival ; 
}

const char* BSys::getenvvar( const char* envprefix, const char* envkey, const char* fallback )
{
    char envvar[128];
    snprintf(envvar, 128, "%s%s", envprefix, envkey );
    const char* evalue = getenv(envvar);
    return evalue ? evalue : fallback ; 
}

int BSys::setenvvar( const char* envprefix, const char* key, const char* value, bool overwrite)
{
    // heap as putenv does not copy

    char* ekey = new char[128] ;
    snprintf(ekey, 128, "%s%s", envprefix, key );

    char* ekv = new char[256] ;  
    snprintf(ekv, 256, "%s=%s", ekey, value );

    const char* prior = getenv(ekey) ;

    int rc = ( overwrite || !prior ) ? putenv(ekv) : 0  ; 

    const char* after = getenv(ekey) ;

    LOG(trace) << "BSys::setenvvar"
              << " ekey " << ekey 
              << " overwrite " << overwrite
              << " prior " << ( prior ? prior : "NULL" )
              << " value " << ( value ? value : "NULL" )   
              << " after " << ( after ? after : "NULL" )   
              << " rc " << rc 
              ;

    return rc ;
} 




