#include "BTimesTable.hh"

#include "OPTICKS_LOG.hh"


void load_dump()
{
    LOG(info) ; 
    const char* dir = "$TMP/evt/dayabay/torch/1" ; 
    BTimesTable* tt = new BTimesTable("t_absolute,t_delta"); 
    tt->load(dir);  // attempts to load t_absolute.ini and t_delta.ini from the directory : corresponding to table columns
    tt->dump();
}

void quad_add()
{
    LOG(info) ; 
    const char* columns = "A,B,C,D" ; 

    BTimesTable* tt = new BTimesTable(columns); 

    for(unsigned i=0 ; i < 20 ; i++) tt->add(i, i*0, i*10, i*20, i*30 );
    tt->add("hello", 42, 42, 42, 42 );

    const char* check = "check" ;
    tt->add(check, 43, 43, 43, 43 );

    tt->dump();
    const char* dir = "$TMP/BTimesTableTest/quad_add" ;
    tt->save(dir) ; 

    BTimesTable* zz = new BTimesTable(columns); 
    zz->load(dir);
    zz->dump();
}

void filter_dump()
{
    LOG(info) ; 
    const char* columns = "A,B,C,D" ; 
    BTimesTable* tt = new BTimesTable(columns); 

    tt->add("red", 0, 10, 20, 30, 0 );
    tt->add("red", 0, 10, 20, 30, 1 );
    tt->add("red", 0, 10, 20, 30, 2 );

    tt->add("gred", 0, 10, 20, 30, 0 );
    tt->add("gred", 0, 10, 20, 30, 1 );
    tt->add("gred", 0, 10, 20, 30, 2 );
   
    tt->add("rouge", 0, 10, 20, 30, 0 );
    tt->add("rouge", 0, 10, 20, 30, 1 );
    tt->add("rouge", 0, 10, 20, 30, 2 );
     
    tt->add("rout", 0, 10, 20, 30, 0 );
    tt->add("rout", 0, 10, 20, 30, 1 );
    tt->add("rout", 0, 10, 20, 30, 2 );

    tt->add("Opticks::Opticks", 0, 10, 20, 30, 0 );
    tt->add("OPropagator::launch", 0, 10, 20, 30, 0 );
    tt->add("OPropagator::launch", 0, 10, 20, 30, 1 );
    

    tt->dump("unfiltered");

    tt->dump("starting with ro", "ro");
 
    tt->dump("starting with OPropagator::launch", "OPropagator::launch");
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    load_dump();
    quad_add();
    filter_dump();

    return 0 ; 
}
