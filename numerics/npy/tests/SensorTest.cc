#include "Sensor.hpp"
#include "stdlib.h"
#include "stdio.h"
#include "assert.h"

int main(int argc, char** argv)
{
    char* idpath = getenv("IDPATH");
    if(!idpath) printf("%s : requires IDPATH envvar \n", argv[0]);

    Sensor sensor;
    sensor.load(idpath);

    return 0 ;
}
