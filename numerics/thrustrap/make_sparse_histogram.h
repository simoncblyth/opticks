#pragma once

typedef unsigned long long History_t ;

#ifdef SPARTAN
#include "Flags.hh"
void make_sparse_histogram(History_t* data, unsigned int numElements, Flags* flags );
#else
#include "Types.hpp"
void make_sparse_histogram(History_t* data, unsigned int numElements, Types* flags );
#endif




