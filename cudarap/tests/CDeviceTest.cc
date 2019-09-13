#include <vector>
#include "CDevice.hh"
#include "OPTICKS_LOG.hh"

void test_SaveLoad(const char* dirpath)
{
    std::vector<CDevice> devs, devs2 ; 

    bool nosave = true ; 

    CDevice::Visible(devs, dirpath, nosave ); 
    CDevice::Dump( devs ); 

    CDevice::Save(devs, dirpath ); 
    CDevice::Load(devs2, dirpath ); 
    CDevice::Dump( devs2 ); 
}

void test_Visible(const char* dirpath)
{
    bool nosave = false ; 
    std::vector<CDevice> devs ; 
    CDevice::Visible(devs, dirpath, nosave ); 
    CDevice::Dump( devs ); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* dirpath = "/tmp"  ; 

    //test_SaveLoad(dirpath); 
    test_Visible(dirpath); 

    return 0 ; 
}


