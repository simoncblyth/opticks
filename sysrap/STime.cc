#include "STime.hh"
#include <time.h>

int STime::EpochSeconds()
{
    time_t now = time(0);
    return now ; 
}


const char* STime::FMT = "%Y%m%d_%H%M%S" ; 
 
std::string STime::Format(int epochseconds, const char* fmt)
{
    const char* ufmt = fmt == NULL ? FMT : fmt ;  

    int t = epochseconds == 0 ? EpochSeconds() : epochseconds ; 
    time_t now(t) ;  
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), ufmt, &tstruct);
    return buf ;
}


