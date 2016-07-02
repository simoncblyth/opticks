#include "TimesTable.hpp"

int main()
{
    const char* dir = "$LOCAL_BASE/env/opticks/rainbow/mdtorch/5/20151226_154520" ;

    TimesTable* tt = new TimesTable("t_absolute,t_delta"); 
    tt->load(dir);
    tt->dump();

    return 0 ; 
}
