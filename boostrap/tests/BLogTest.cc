#include "OPTICKS_LOG.hh"
#include "BLog.hh"
#include "BStr.hh"
#include "BTxt.hh"
#include "BFile.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    int pindex = argc > 1 ? BStr::atoi(argv[1]) : 1872 ; 

    const char* logpath = BStr::concat<int>("$TMP/ox_", pindex, ".log") ; 
    const char* txtpath = BStr::concat<int>("$TMP/ox_", pindex, ".txt") ; 

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
