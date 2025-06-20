/**
sysrap/tests/sseq_record_test.sh
=================================

::

    ~/opticks/sysrap/tests/sseq_record_test.sh

**/

#include "sseq_record.h"
#include "ssys.h"

struct sseq_record_test
{
    static int LoadRecordSeqSelection();
    static int Main();
};


int sseq_record_test::LoadRecordSeqSelection()
{
    std::cout << "[LoadRecordSeqSelection\n";
    const char* _fold = "$AFOLD" ;
    const char* _seqhis = "${AFOLD_RECORD_SLICE:-TO BT BT BT BT BR BT BT BT BT BT BT SC BT BT BT BT SD}" ;

    NP* a = sseq_record::LoadRecordSeqSelection(_fold, _seqhis);

    std::cout << " a " << ( a ? a->sstr() : "-" ) << "\n" ;
    std::cout << "]LoadRecordSeqSelection\n";
    return 0;
}

int sseq_record_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "LoadRecordSeqSelection" );
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"LoadRecordSeqSelection")) rc += LoadRecordSeqSelection();

    return rc ;
}


int main()
{
    return sseq_record_test::Main() ;
}


