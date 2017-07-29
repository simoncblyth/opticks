#pragma once

void fswap_ptr(float* a, float* b)
{
    float tmp = *a;
    *a = *b;
    *b = tmp;
}
float* fmin_ptr(float* a, float* b)
{
    return *a < *b ? a : b  ;  
}
void fascending_ptr(unsigned num, float* a )
{
    if(num == 3)
    {
        fswap_ptr( &a[0], fmin_ptr( &a[0], fmin_ptr(&a[1], &a[2])));
        fswap_ptr( &a[1], fmin_ptr( &a[1], &a[2] ) );
    }
    else if( num == 2 )
    {
        fswap_ptr( &a[0], fmin_ptr( &a[0], &a[1] ));
    }
    else if(num == 4)
    {
        fswap_ptr( &a[0], fmin_ptr( &a[0], fmin_ptr( &a[1], fmin_ptr(&a[2], &a[3]))));
        fswap_ptr( &a[1], fmin_ptr( &a[1], fmin_ptr( &a[2], &a[3])));
        fswap_ptr( &a[2], fmin_ptr( &a[2], &a[3] ) );
    }
}


