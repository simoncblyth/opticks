/**
~/o/sysrap/tests/sFFS_test.sh
**/

#include <iostream>
#include <iomanip>
#include <cassert>

#include "sFFS.h"


void dump()
{
    for(unsigned i=0 ; i < 32 ; i++)
    {
         unsigned flag = 0x1 << i ;
         unsigned FFS_flag = FFS(flag);

         std::cout
             << std::setw(3) << i
             << std::setw(12) << flag
             << std::setw(12) << std::hex << flag << std::dec
             << std::setw(12) << FFS_flag
             << "\n"
             ;
    }
}



int main()
{
    dump();

    std::cout << "Starting sFFS_test..." << std::endl;

    // --- Test 32-bit FFS & rFFS ---
    assert(FFS(0) == 0);
    assert(FFS(1) == 1);     // bit 1 -> index 1
    assert(FFS(2) == 2);     // bit 2 -> index 2
    assert(FFS(4) == 3);     // bit 3 -> index 3
    assert(FFS(0x80000000) == 32);

    assert(rFFS(0) == 0);
    assert(rFFS(1) == 1);
    assert(rFFS(2) == 2);
    assert(rFFS(3) == 4);
    assert(rFFS(32) == 0x80000000);

    // --- Test 64-bit FFSLL & rFFSLL ---
    assert(FFSLL(0ULL) == 0);
    assert(FFSLL(1ULL) == 1);
    assert(FFSLL(0x80000000ULL) == 32);
    assert(FFSLL(0x100000000ULL) == 33);             // Exceeds 32-bit limit
    assert(FFSLL(0x8000000000000000ULL) == 64);      // Highest bit in 64-bit integer

    assert(rFFSLL(0) == 0);
    assert(rFFSLL(1) == 1);
    assert(rFFSLL(32) == 0x80000000ULL);
    assert(rFFSLL(33) == 0x100000000ULL);
    assert(rFFSLL(64) == 0x8000000000000000ULL);

    // --- Round-trip consistency check ---
    for (unsigned int i = 1; i <= 32; ++i) {
        assert(FFS(rFFS(i)) == static_cast<int>(i));
    }
    for (unsigned long long i = 1; i <= 64; ++i) {
        assert(FFSLL(rFFSLL(i)) == static_cast<int>(i));
    }

    std::cout << "ALL TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}

