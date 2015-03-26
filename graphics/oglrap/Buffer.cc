#include "Buffer.hh"

#include "stdio.h"

void Buffer::Summary(const char* msg)
{
    printf("%s NumBytes %u Pointer %p \n", msg, getNumBytes(), getPointer());
}
