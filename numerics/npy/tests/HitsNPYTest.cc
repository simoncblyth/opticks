#include "HitsNPY.hpp"
#include "NSensorList.hpp"
#include "NSensor.hpp"

#include "stdlib.h"
#include "stdio.h"
#include "assert.h"

int main(int argc, char** argv)
{
    char* idpath = getenv("IDPATH");
    if(!idpath) printf("%s : requires IDPATH envvar \n", argv[0]);

    NSensorList sens;
    sens.load(idpath, "idmap");

    const char* tag = "1" ; 
    NPY<float>* photons = NPY<float>::load("oxtorch", tag,"dayabay");
    

    HitsNPY hits(photons, &sens );
    hits.debugdump();




    return 0 ;
}


