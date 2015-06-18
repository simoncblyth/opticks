#include "GSensorList.hh"
#include "stdlib.h"
#include "stdio.h"
#include "assert.h"

int main(int argc, char** argv)
{
    char* idpath = getenv("IDPATH");
    if(!idpath) printf("%s : requires IDPATH envvar \n", argv[0]);

    GSensorList sens;
    sens.load(idpath, "idmap");

    if(getenv("VERBOSE")) sens.dump();


    return 0 ;
}
