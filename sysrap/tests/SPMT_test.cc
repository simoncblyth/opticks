#include "SPMT.h"

int main(int argc, char** argv)
{
    SPMT* pmt = SPMT::CreateFromJPMT();
    if(pmt == nullptr) return 1 ;

    std::cout << "SPMT_test.main " << pmt->desc() << "\n" ;
    NPFold* spmt_f = pmt->serialize();
    spmt_f->save("$FOLD/spmt");

    NPFold* testfold = pmt->make_testfold();
    testfold->save("$FOLD/testfold") ;


    return 0 ;
}
