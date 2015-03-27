#include "GBuffer.hh"

#include "stdio.h"

void GBuffer::Summary(const char* msg)
{
    printf("%s NumBytes %u Pointer %p \n", msg, getNumBytes(), getPointer());
}
