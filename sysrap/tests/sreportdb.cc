/**
sreportdb.cc
=============

HMM: for archive dir argument it makes sense to create db inside the archive ?

**/

#include "sreportdb.h"

int main(int argc, char** argv)
{
    std::cout << argv[0] << "\n" ;
    const char* dbpath  = argc > 1 ? argv[1] : "/tmp/sreportdb.db" ;
    const char* input_fold = argc > 2 ? argv[2] : "/report/or/archive/directory" ;

    sreportdb db(dbpath);
    if(db.level > 0) std::cout << db.desc() ;

    db.import_auto(input_fold);
    // import single report or archive of reports depending on directory content

    return 0 ;
}
