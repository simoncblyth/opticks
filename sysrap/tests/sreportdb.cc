/**
sreportdb.cc - creates sqlite3 DB and populates tables from persisted sreport folders
======================================================================================

~/o/sysrap/tests/sreportdb.sh

For how this fits into the monitoring workflow see:

~/o/sysrap/sreportdb_monitoring_workflow.rst

**/

#include "sreportdb.h"

int main(int argc, char** argv)
{
    const char* dbfold  = argc > 1 ? argv[1] : "/tmp" ;
    const char* infold  = argc > 2 ? argv[2] : "/report/or/archive/directory" ;

    std::cout
        << argv[0]
        << " dbfold [" << ( dbfold ? dbfold : "-" ) << "]"
        << " infold [" << ( infold ? infold : "-" ) << "]"
        << "\n"
        ;

    sreportdb db(dbfold);
    if(db.level > 0) std::cout << db.desc() ;

    db.import_auto(infold);
    // import single report or archive of reports depending on directory content

    static const char* _sreportdb__dupe_import = "sreportdb__dupe_import" ;
    int sreportdb__dupe_import = U::GetEnvInt(_sreportdb__dupe_import, 0) ; // dupe detection should prevent the imports
    std::cout << " [" << _sreportdb__dupe_import << "]: " << ( sreportdb__dupe_import > 0 ? "YES" : "NO " ) << "\n" ;

    if(sreportdb__dupe_import > 0)
    {
        db.import_auto(infold);
        db.import_auto(infold);
        db.import_auto(infold);
        db.import_auto(infold);
    }

    std::cout
        << "[sreportdb.desc_runs\n"
        << db.desc_runs()
        << "]sreportdb.desc_runs\n"
        ;

    return 0 ;
}
