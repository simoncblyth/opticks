/**

TEST=OpticksTwoTest om-t

while true; do OpticksTwoTest ; done
while OpticksTwoTest ; do echo -n ; done 

**/

#include "OPTICKS_LOG.hh"

#include "Opticks.hh"
#include "OpticksQuery.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
 
    Opticks ok(argc, argv);
    ok.configure();
    OpticksQuery* q = ok.getQuery();

    LOG(info)  << "idpath " << ok.getIdPath() ;
    LOG(info) << "q\n" <<  q->desc() ; 

    const char* key = "CX4GDMLTest.X4PhysicalVolume.World0xc15cfc0_PV.27c39be4e46a36ea28a3c4da52522c9e" ; 
    Opticks::SetKey(key);


    //Opticks ok1(0,0);
    Opticks ok1(argc,argv);
    ok1.configure();
    OpticksQuery* q1 = ok1.getQuery();

    LOG(info)  << "idpath1 " << ok1.getIdPath() ;
    LOG(info) << "q1\n" <<  ( q1 ? q1->desc() : "NULL-query" ) ; 


    return 0 ;   
}
