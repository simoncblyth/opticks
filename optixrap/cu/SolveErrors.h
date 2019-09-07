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



#ifndef __CUDACC__
void cubic_errors(Solve_t a, Solve_t b, Solve_t c, Solve_t* rts, Solve_t* rterr, Solve_t* rtdel, int nrts)
{
    Solve_t nought = 0.f ; 
    Solve_t doub4 = 4.f ; 
    Solve_t doub3 = 3.f ; 
    Solve_t doub2 = 2.f ; 
    Solve_t doub12 = 12.f ; 
    Solve_t doub6 = 6.f ; 
    Solve_t doub24 = 24.f ; 
/*

In [24]: ex = x**3 + a*x**2 + b*x + c

In [25]: diff(ex,x)
Out[25]: 2*a*x + b + 3*x**2

In [26]: diff(diff(ex,x),x)
Out[26]: 2*a + 6*x

In [27]: diff(diff(diff(ex,x),x),x)
Out[27]: 6
    
*/ 
    for ( int k = 0 ; k < nrts ; ++ k ) 
    {   
        rtdel[k] = ((rts[k]+a)*rts[k]+b)*rts[k]+c ;

        if (rtdel[k] == nought) 
        { 
            rterr[k] = nought;
        }
        else
        {   
            Solve_t deriv = (doub3*rts[k]+doub2*a)*rts[k]+b  ;
            if (deriv != nought)
            {
                rterr[k] = fabs(rtdel[k]/deriv);
            }
            else
            {   
               deriv = doub6*rts[k]+doub2*a  ;
               if (deriv != nought)
               {
                   rterr[k] = sqrt(fabs(rtdel[k]/deriv)) ;
               }
            }   
         }   
      }   
}


void quartic_errors(Solve_t a,Solve_t b,Solve_t c,Solve_t d, Solve_t* rts,Solve_t* rterr, Solve_t* rtdel,int nrts)
{
    Solve_t nought = 0.f ; 
    Solve_t doub4 = 4.f ; 
    Solve_t doub3 = 3.f ; 
    Solve_t doub2 = 2.f ; 
    Solve_t doub12 = 12.f ; 
    Solve_t doub6 = 6.f ; 
    Solve_t doub24 = 24.f ; 
 
    /*

In [18]: ex
Out[18]: a*x**3 + b*x**2 + c*x + d + x**4

In [19]: diff(ex,x)
Out[19]: 3*a*x**2 + 2*b*x + c + 4*x**3

In [20]: diff(diff(ex,x),x)
Out[20]: 6*a*x + 2*b + 12*x**2

In [21]: diff(diff(diff(ex,x),x),x)
Out[21]: 6*a + 24*x

In [22]: diff(diff(diff(diff(ex,x),x),x),x)
Out[22]: 24

     
    */

    for ( int k = 0 ; k < nrts ; ++ k ) 
    {   
        rtdel[k] = (((rts[k]+a)*rts[k]+b)*rts[k]+c)*rts[k]+d ;

        if (rtdel[k] == nought) 
        { 
            rterr[k] = nought;
        }
        else
        {   
            Solve_t deriv = ((doub4*rts[k]+doub3*a)*rts[k]+doub2*b)*rts[k]+c ;
            if (deriv != nought)
            {
                rterr[k] = fabs(rtdel[k]/deriv);
            }
            else
            {   
               deriv = (doub12*rts[k]+doub6*a)*rts[k]+doub2*b ;
               if (deriv != nought)
               {
                   rterr[k] = sqrt(fabs(rtdel[k]/deriv)) ;
               }
               else
               {   
                   deriv = doub24*rts[k]+doub6*a ;
                   rterr[k] = deriv != nought ?   cbrt(fabs(rtdel[k]/deriv)) : sqrt(sqrt(fabs(rtdel[k])/doub24)) ; 
               }   
            }   
         }   
      }   
} 

#endif




