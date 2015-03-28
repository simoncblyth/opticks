#include "GBuffer.hh"

#include "stdio.h"

void GBuffer::Summary(const char* msg)
{
    printf("%s NumBytes %u Pointer %p ItemSize %u  NumElements_per_item %u NumItems %u \n", msg, getNumBytes(), getPointer(), getItemSize(), getNumElements(), getNumItems() );
}

