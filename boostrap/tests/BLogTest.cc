#include "OPTICKS_LOG.hh"
#include "BLog.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    BLog* ucf = BLog::Load("$TMP/ox_1872.log"); 
    ucf->dump("ucf"); 


    return 0 ; 
}
