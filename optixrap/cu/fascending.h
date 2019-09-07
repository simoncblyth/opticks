/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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


