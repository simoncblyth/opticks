#include "TimesTable.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


void load_dump()
{
    const char* dir = "$TMP/evt/dayabay/torch/1" ; 
    TimesTable* tt = new TimesTable("t_absolute,t_delta"); 
    tt->load(dir);
    tt->dump();
}

void quad_add()
{
    const char* columns = "A,B,C,D" ; 

    TimesTable* tt = new TimesTable(columns); 

    for(unsigned i=0 ; i < 20 ; i++) tt->add(i, i*0, i*10, i*20, i*30 );
    tt->add("hello", 42, 42, 42, 42 );

    const char* check = "check" ;
    tt->add(check, 43, 43, 43, 43 );



    tt->dump();
    const char* dir = "$TMP/TimesTableTest/quad_add" ;
    tt->save(dir) ; 

    TimesTable* zz = new TimesTable(columns); 
    zz->load(dir);
    zz->dump();
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    NPY_LOG__ ; 

    load_dump();
    quad_add();


    return 0 ; 
}
