// hmm see ggeo/tests/GAttrSeqTest.cc

#include "Opticks.hh"
#include "OpticksAttrSeq.hh"
#include "Index.hpp"

#include "OKCORE_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OKCORE_LOG__ ;   

    Opticks ok(argc, argv);
    ok.configure();

    Index* i = ok.loadHistoryIndex();

    if(i == NULL) LOG(info) << "try eg \"--cat concentric\" option " 


    return 0 ; 
}
