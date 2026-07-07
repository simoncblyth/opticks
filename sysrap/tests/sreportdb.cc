/**
sreportdb.cc
=============

~/o/sysrap/tests/sreportdb.sh

HMM: for archive dir argument does it makes sense to create db inside the archive ?

* need to consider gitlab-ci practicalities

**/

#include "sreportdb.h"

int main(int argc, char** argv)
{
    std::cout << argv[0] << "\n" ;
    const char* dbfold  = argc > 1 ? argv[1] : "/tmp" ;
    const char* infold  = argc > 2 ? argv[2] : "/report/or/archive/directory" ;

    sreportdb db(dbfold);
    if(db.level > 0) std::cout << db.desc() ;

    db.import_auto(infold);
    // import single report or archive of reports depending on directory content

    return 0 ;
}
