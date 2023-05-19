// ./stamp_test.sh 
#include "stamp.h"
#include "NP.hh"

int main()
{
    static const int N = 1000000*25 ;  
    // old laptop: counting and stamp recording to 25M takes roughly 1s (1M us)
    // so 25 stamp records takes ~1us 
 
    NP* t = NP::Make<uint64_t>(N) ;  
    uint64_t* tt = t->values<uint64_t>(); 
    for(int i=0 ; i < N ; i++) tt[i] = stamp::Now(); 
    t->save("$TTPATH");  

    return 0 ; 
}

