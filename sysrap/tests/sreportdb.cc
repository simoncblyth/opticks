#include "sreportdb.h"

int main(int argc, char** argv)
{

    std::cout << argv[0] << "\n" ;
    const char* dbpath  = argc > 1 ? argv[1] : "/tmp/sreportdb.db" ;
    const char* runfold = argc > 2 ? argv[2] : "/directory/with/run/and/evsmry/arrays" ;

    sreportdb db(dbpath);
    if(db.level > 0) std::cout << db.desc() ;

    db.import_run(runfold);

    return 0 ;
}
