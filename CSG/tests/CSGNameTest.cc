#include "SSys.hh"
#include "SName.h"

#include "OPTICKS_LOG.hh"
#include "scuda.h"
#include "CSGFoundry.h"


void test_getNumName( const SName* id )
{
    LOG(info) << " id.getNumName " << id->getNumName() ; 
}

void test_getName( const SName* id )
{
    for(unsigned i=0 ; i < id->getNumName() ; i++)
    {
         const char* name = id->getName(i); 
         LOG(info) << std::setw(4) << i << " : " << name ; 
    }
}



void test_findIndex( const SName* id )
{
    for(unsigned i=0 ; i < id->getNumName() ; i++)
    {
         const char* name = id->getName(i); 
         unsigned count = 0 ; 
         int idx = id->findIndex(name, count); 
         char msg = idx == int(i) && count == 1 ? ' ' : '*' ; 
         LOG(info) 
             << " i " << std::setw(4) << i 
             << " idx " << std::setw(4) << idx 
             << " count " << std::setw(4) << count
             << " "
             << msg
             << " "
             << name 
             ; 

    }
}


void test_getAbbr( const SName* id )
{
    for(unsigned i=0 ; i < id->getNumName() ; i++)
    {
         const char* name = id->getName(i); 
         const char* abbr = id->getAbbr(i); 
         LOG(info) 
             << std::setw(4) << i 
             << " "
             << std::setw(50) << abbr
             << " : "
             << std::setw(50) << name
             ; 

    }
}

void test_parseArg( const SName* id, const char* arg )
{
    unsigned count = 0 ; 
    int idx = id->parseArg(arg, count);
    const char* name = idx > -1 ? id->getName(idx) : nullptr ; 
    LOG(info)
       << std::setw(20) << arg
       << " : " 
       << std::setw(4) << idx
       << " : "
       << std::setw(4) << count
       << " : "
       << name
       ;
}

void test_parseArg( const SName* id, int argc, char** argv )
{
    std::vector<std::string> args = {"uni", "130", "sWat" } ; 
    for(int i=0 ; i < int(args.size()) ; i++) test_parseArg(id, args[i].c_str()) ;
    for(int i=1 ; i < argc             ; i++) test_parseArg(id, argv[i]);  
}

void test_parseMOI( const SName* id, const char* moi )
{
    int midx, mord, iidx ; 
    id->parseMOI(midx, mord, iidx, moi);  
    const char* name = midx > -1 ? id->getName(midx) : nullptr ; 

    LOG(info)
       << std::setw(20) << moi
       << " : " 
       << std::setw(4) << midx
       << " : "
       << std::setw(4) << mord
       << " : "
       << std::setw(4) << iidx
       << " : "
       << name
       ;
}

void test_parseMOI( const SName* id, int argc, char** argv )
{
    std::vector<std::string> args = {"uni", "130", "sWat" } ; 
    for(int i=0 ; i < int(args.size()) ; i++) test_parseMOI(id, args[i].c_str()) ;
    for(int i=1 ; i < argc             ; i++) test_parseMOI(id, argv[i]);  
}

void test_parseMOI_fd( const CSGFoundry* fd )
{
    const char* moi = SSys::getenvvar("MOI", "sWorld:0:0"); 
    int midx, mord, iidx ; 
    fd->parseMOI(midx, mord, iidx,  moi );  

    LOG(info) 
        << " MOI " << moi 
        << " midx " << midx 
        << " mord " << mord 
        << " iidx " << iidx
        ;   
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGFoundry* fd = CSGFoundry::Load(); 
    LOG(info) << "foundry " << fd->desc() ; 
    fd->summary(); 

    const SName* id = fd->id ; 
    LOG(info) << id->desc(); 

    LOG(info) << "[ descMeshPrim " ;  
    LOG(info) << fd->descMeshPrim() ; 
    LOG(info) << "] descMeshPrim " ;  

    //test_getNumName(id); 
    //test_getName(id); 
    //test_findIndex(id); 
    //test_getAbbr(id); 
    //test_parseArg(id, argc, argv); 
    //test_parseMOI(id, argc, argv); 
    //test_parseMOI_fd(fd); 

    return 0 ; 
}



