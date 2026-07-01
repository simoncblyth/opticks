/**
sfilesystem_test.cc
====================

~/o/sysrap/tests/sfilesystem_test.sh

**/

#include "ssys.h"
#include "sfilesystem.h"


struct sfilesystem_test
{
    static int find_index_of_max_indexed_dirname();
    static int ExecutablePath();
    static int ExecutablePathSibling();
    static int Main();
};


inline int sfilesystem_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL");
    bool ALL = strcmp(TEST, "ALL") == 0 ;
    int rc = 0 ;

    if(ALL||0==strcmp(TEST,"find_index_of_max_indexed_dirname")) rc += find_index_of_max_indexed_dirname();
    if(ALL||0==strcmp(TEST,"ExecutablePath")) rc += ExecutablePath();
    if(ALL||0==strcmp(TEST,"ExecutablePathSibling")) rc += ExecutablePathSibling();

    return rc ;

}

inline int sfilesystem_test::find_index_of_max_indexed_dirname()
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

inline int sfilesystem_test::ExecutablePath()
{
    std::string bin = sfilesystem::ExecutablePath();
    std::cout << " bin " << bin << "\n" ;
    return 0 ;
}

inline int sfilesystem_test::ExecutablePathSibling()
{
    const char* sibling = "sreportdb.sql" ;
    std::string sib = sfilesystem::ExecutablePathSibling(sibling);
    std::cout << " sib " << sib << "\n" ;
    return 0 ;
}


int main()
{
    return sfilesystem_test::Main();
}
