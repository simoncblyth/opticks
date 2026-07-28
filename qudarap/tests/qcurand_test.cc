#include <iostream>
#include <vector>
#include "qcurand_test.h"
#include "NPFold.h"

template <typename T>
NP* run_high_stats_test(size_t num_values, unsigned long seed)
{
    NP* a = NP::Make<T>(num_values);
    run_qcurand_test<T>(a->values<T>(), num_values, seed);
    return a ;
}

int main()
{
    size_t num_values = 10000000; // 10M samples
    unsigned long seed = 1234ULL;
    NPFold* fold = new NPFold ;
    fold->add("f", run_high_stats_test<float>( num_values, seed) );
    fold->add("d", run_high_stats_test<double>(num_values, seed) );
    fold->save("$FOLD");
    return 0;
}




