#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    plog::Severity level = info ; 

    pLOG(level,4)  << " hello+4 " ; 
    pLOG(level,3)  << " hello+3 " ; 
    pLOG(level,2)  << " hello+2 " ; 
    pLOG(level,1)  << " hello+1 " ; 
    pLOG(level,0)  << " hello+0 " ; 
    pLOG(level,-1) << " hello-1 " ; 
    pLOG(level,-2) << " hello-2 " ; 
    pLOG(level,-3) << " hello-3 " ; 
    pLOG(level,-4) << " hello-4 " ; 

/*

2018-08-04 09:44:56.320 VERB  [8369891] [main@14]  hello+4 
2018-08-04 09:44:56.320 VERB  [8369891] [main@15]  hello+3 
2018-08-04 09:44:56.320 VERB  [8369891] [main@16]  hello+2 
2018-08-04 09:44:56.320 DEBUG [8369891] [main@17]  hello+1 
2018-08-04 09:44:56.320 INFO  [8369891] [main@18]  hello+0 
2018-08-04 09:44:56.320 WARN  [8369891] [main@19]  hello-1 
2018-08-04 09:44:56.320 ERROR [8369891] [main@20]  hello-2 
2018-08-04 09:44:56.320 FATAL [8369891] [main@21]  hello-3 
2018-08-04 09:44:56.320 FATAL [8369891] [main@22]  hello-4 

*/

    return 0 ; 
}

