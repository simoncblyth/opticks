#include "OPTICKS_LOG.hh"
#include "BLog.hh"
#include "BTxt.hh"
#include "BFile.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* logpath = "$TMP/ox_1872.log" ; 
    const char* txtpath = "$TMP/ox_1872.txt" ; 

    BLog* a = BLog::Load(logpath); 
    const std::vector<double>&  av = a->getValues() ; 
    a->setSequence(&av) ; 
    a->dump("a"); 
    a->write(txtpath); 

    BLog* b = BLog::Load(txtpath); 
    b->dump("b"); 

    int RC = BLog::Compare(a, b ); 
    assert( RC == 0 ) ; 


    return 0 ; 
}
