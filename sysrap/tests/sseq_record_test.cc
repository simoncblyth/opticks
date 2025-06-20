/**
sysrap/tests/sseq_record_test.sh
=================================

::

    ~/opticks/sysrap/tests/sseq_record_test.sh

TODO: save selected records into AFOLD with appropriate name eg "record_TO_BT_AB.npy"

**/

#include "sseq_record.h"
#include "ssys.h"

int main()
{
    sseq_record* a = sseq_record::Load("$AFOLD");
    const char* q_startswith = ssys::getenvvar("Q_STARTSWITH", "TO BT AB") ;
    NP* record_sel = a->create_record_selection(q_startswith);
    std::cout
        << "record_sel " << record_sel->sstr() << "\n"
        ;

    return 0 ;
}


