
#include <string>
#include <cstring>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <iostream>

#include "PLOG.hh"
#include "STranche.hh"


STranche::STranche(unsigned total_, unsigned max_tranche_) 
    :
    total(total_),
    max_tranche(max_tranche_),
    num_tranche((total+max_tranche-1)/max_tranche),
    last_tranche(total - (num_tranche-1)*max_tranche)  // <-- is max_tranche when  total % max_tranche == 0 
{
}

unsigned STranche::tranche_size(unsigned i) const 
{
    assert( i < num_tranche && " trance indices must be from 0 to tr.num_tranche - 1 inclusive  " ); 
    return i < num_tranche - 1 ? max_tranche : last_tranche  ; 
}
unsigned STranche::global_index(unsigned i, unsigned j ) const 
{
    return max_tranche*i + j ; 
}


const char* STranche::desc() const 
{
    std::stringstream ss ; 

    ss << "STranche"
       << " total " << total 
       << " max_tranche " << max_tranche 
       << " num_tranche " << num_tranche 
       << " last_tranche " << last_tranche 
       ;

    std::string s = ss.str();
    return strdup(s.c_str());
}


void STranche::dump(const char* msg)
{
    LOG(info) << msg << " desc " << desc() ; 

    unsigned cumsum = 0 ; 
    for(unsigned i=0 ; i < num_tranche ; i++)
    {
         unsigned size = tranche_size(i) ; 
         cumsum += size ; 

         unsigned global_index_0 = global_index(i, 0);
         unsigned global_index_1 = global_index(i, size-1); 

         std::cout << " i " << std::setw(6) << i 
                   << " tranche_size " << std::setw(6) << size
                   << " global_index_0 " << std::setw(6) << global_index_0
                   << " global_index_1 " << std::setw(6) << global_index_1
                   << " cumsum " << std::setw(6) << cumsum
                   << std::endl 
                   ;

         assert( cumsum == global_index_1 + 1 );
    }
    assert( cumsum == total ); 
}



