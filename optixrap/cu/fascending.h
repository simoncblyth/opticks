#pragma once

void fswap_ptr(Solve_t* a, Solve_t* b)
{
    Solve_t tmp = *a;
    *a = *b;
    *b = tmp;
}
Solve_t* fmin_ptr(Solve_t* a, Solve_t* b)
{
    return *a < *b ? a : b  ;  
}


void fascending_ptr(unsigned num, Solve_t* a )
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


