#include <vector>
#include "CDevice.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv); 
    ok.configure(); 

    std::vector<CDevice> devs ; 

    const char* dirpath = nullptr ; 
    bool nosave = true ; 
    CDevice::Visible(devs, dirpath, nosave ); 
    CDevice::Dump( devs , "visible devices"); 

    LOG(info) <<  "CDevice::VISIBLE_COUNT " << CDevice::VISIBLE_COUNT ; 
 
    return 0 ; 
}


/*

epsilon:thrustrap blyth$ TDeviceDumpTest 
2021-06-07 16:37:20.304 INFO  [117662] [CDevice::Dump@265] visible devices[0:GeForce_GT_750M]
2021-06-07 16:37:20.304 INFO  [117662] [CDevice::Dump@269] idx/ord/mpc/cc:0/0/2/30   2.000 GB  GeForce GT 750M
2021-06-07 16:37:20.304 INFO  [117662] [main@20] CDevice::VISIBLE_COUNT 1

epsilon:thrustrap blyth$ TDeviceDumpTest --cvd -
2021-06-07 16:37:31.044 ERROR [117740] [Opticks::postconfigureCVD@3000]  --cvd [-] option internally sets CUDA_VISIBLE_DEVICES []
2021-06-07 16:37:31.943 ERROR [117740] [CDevice::Load@326]  failed read from CDevice.bin
2021-06-07 16:37:31.943 INFO  [117740] [CDevice::Dump@265] visible devices[]
2021-06-07 16:37:31.943 INFO  [117740] [main@20] CDevice::VISIBLE_COUNT 0

epsilon:thrustrap blyth$ CVD=- TDeviceDumpTest 
2021-06-07 16:37:39.771 ERROR [117796] [Opticks::postconfigureCVD@3000]  --cvd [-] option internally sets CUDA_VISIBLE_DEVICES []
2021-06-07 16:37:40.511 ERROR [117796] [CDevice::Load@326]  failed read from CDevice.bin
2021-06-07 16:37:40.511 INFO  [117796] [CDevice::Dump@265] visible devices[]
2021-06-07 16:37:40.511 INFO  [117796] [main@20] CDevice::VISIBLE_COUNT 0

*/


