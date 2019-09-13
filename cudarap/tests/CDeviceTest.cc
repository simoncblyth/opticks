#include <vector>
#include "CDevice.hh"
#include "OPTICKS_LOG.hh"

void test_SaveLoad(const char* dirpath)
{
    std::vector<CDevice> devs, devs2 ; 

    bool nosave = true ; 
    CDevice::Visible(devs, dirpath, nosave ); 
    CDevice::Dump( devs, "visible devices"); 

    CDevice::Save(devs, dirpath ); 
    CDevice::Load(devs2, dirpath ); 
    CDevice::Dump( devs2, "loaded devs2"); 
}

void test_Visible(const char* dirpath)
{
    bool nosave = false ; 
    std::vector<CDevice> devs ; 
    CDevice::Visible(devs, dirpath, nosave ); 
    CDevice::Dump( devs , "visible devices"); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //const char* dirpath = "/tmp"  ; 
    const char* dirpath = "/home/blyth/.opticks/runcache" ; 

    //test_SaveLoad(dirpath); 
    test_Visible(dirpath); 

    return 0 ; 
}


