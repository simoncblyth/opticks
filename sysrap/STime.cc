#include "STime.hh"
#include <time.h>

int STime::EpochSeconds()
{
    time_t now = time(0);
    return now ; 
}
 
std::string STime::Format(const char* fmt, int epochseconds)
{
    int t = epochseconds == 0 ? EpochSeconds() : epochseconds ; 
    time_t now(t) ;  
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), fmt, &tstruct);
    return buf ;
}


