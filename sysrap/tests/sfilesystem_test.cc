/**
sfilesystem_test.cc
====================

~/o/sysrap/tests/sfilesystem_test.sh

**/

#include "sfilesystem.h"

int main()
{
    const char* container = "container_dir" ;
    const char* prefix = "sreport_" ;

    long long max_idx = sfilesystem::find_index_of_max_indexed_dirname(container, prefix );
    long long next_idx = max_idx + 1 ;

    std::string next_dirname = sfilesystem::form_indexed_dirname( next_idx, prefix );
    bool is_indexed = sfilesystem::is_indexed_dirname( next_dirname.c_str(), prefix );

    std::cout
         << "sfilesystem_test "
         << " max_idx " << max_idx
         << " next_idx " << next_idx
         << " next_dirname [" << next_dirname << "]"
         << " is_indexed " << ( is_indexed ? "YES" : "NO " )
         << "\n"
         ;

    return 0;
}
