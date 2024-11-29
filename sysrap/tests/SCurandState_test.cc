/**
SCurandState_test.cc
=====================

~/o/sysrap/tests/SCurandState_test.sh 

As an initial goal can try to effectively recreate QCurandState_3000000_0_0.bin
by creation and loading three chunks of 1M each 


**/

#include "ssys.h"
#include "SCurandState.h"


struct SCurandState_test
{
    static int FileStates();
    static int Main();
};


inline int SCurandState_test::FileStates()
{
    long st = _SCurandState::FileStates("QCurandState_1000000_0_0.bin" ); 
    int rc = st == 1000000 ? 0 : 1 ; 
    return rc ; 
}

inline int SCurandState_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "FileStates");
    int rc = 0 ;  
    if(strcmp(TEST,"FileStates") == 0) rc += FileStates(); 
    return rc ; 
}


int main()
{
    return SCurandState_test::Main() ;
}
