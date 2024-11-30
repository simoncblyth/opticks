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
    static int ctor();
    static int NumFromFilesize();
    static int ParseDir();
    static int ChunkLoadSave();
    static int Main();
};


inline int SCurandState_test::ctor()
{
    _SCurandState cs ; 
    std::cout << cs.desc(); 
    return 0 ; 
}

inline int SCurandState_test::NumFromFilesize()
{
    long st = SCurandChunk::NumFromFilesize("QCurandState_1000000_0_0.bin" ); 
    int rc = st == 1000000 ? 0 : 1 ; 
    return rc ; 
}

inline int SCurandState_test::ParseDir()
{
    std::vector<SCurandChunk> chunks ; 
    SCurandChunk::ParseDir(chunks ); 

    return 0 ; 
}

inline int SCurandState_test::ChunkLoadSave()
{
    SCurandChunk chunk = {} ; 
    const char* name0 = "SCurandChunk_0000_1M_0_0.bin" ; 
    const char* name1 = "SCurandChunk_1000_1M_0_0.bin" ; 

    int rc0 = SCurandChunk::Load(chunk, name0 );
    int rc1 = SCurandChunk::Save(chunk, name1 );

    std::cout 
        << "SCurandState_test::ChunkLoadSave" 
        << " rc0 " << rc0 
        << " rc1 " << rc1 
        << "\n" 
        ; 

    return rc0 + rc1 ; 
}



inline int SCurandState_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "NumFromFilesize");
    int rc = 0 ;  
    if(strcmp(TEST,"ctor") == 0)            rc += ctor(); 
    if(strcmp(TEST,"NumFromFilesize") == 0) rc += NumFromFilesize(); 
    if(strcmp(TEST,"ParseDir") == 0)        rc += ParseDir(); 
    if(strcmp(TEST,"ChunkLoadSave") == 0)       rc += ChunkLoadSave(); 
    return rc ; 
}


int main()
{
    return SCurandState_test::Main() ;
}
