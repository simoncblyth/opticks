
#ifdef __CUDACC__
__device__ __host__
#endif
static Solve_t SolveCubicGem(Solve_t p, Solve_t q, Solve_t r)
{
/* 
     find the lowest real root of the cubic - 
       x**3 + p*x**2 + q*x + r = 0 

   input parameters - 
     p,q,r - coeffs of cubic equation. 

   output- 
     cubic - a real root. 

   global constants -
     rt3 - sqrt(3) 
     inv3 - 1/3 
     doubmax - square root of largest number held by machine 

     method - 
     see D.E. Littlewood, "A University Algebra" pp.173 - 6 

     Charles Prineas   April 1981 

     called by  neumark.
     calls  acos3 
*/

   Solve_t rt3 = sqrt(3.f);
   Solve_t doub1 = 1.f ; 
   Solve_t inv2 = 1.f/2.f ; 
   Solve_t inv3 = 1.f/3.f ; 
   Solve_t nought = 0.f ; 

   int nrts = 0 ;
   Solve_t po3,po3sq,qo3;
   Solve_t uo3,u2o3,uo3sq4,uo3cu4 ;
   Solve_t v,vsq,wsq ;
   Solve_t m,mcube,n;
   Solve_t muo3,s,scube,t,cosk,sinsqk ;
   Solve_t root;

//   Solve_t curoot();
//   Solve_t acos3();
//   Solve_t sqrt(),fabs();

   m = nought;

/*
   if ( p > doubmax || p <  -doubmax)  // x**3 + p x**2 + q *x + r = 0 ->   x + p = 0   (p-dominant)
   {
       root = -p;
   }
   else if ( q > doubmax || q <  -doubmax )  //   x**2 = -q   ???  
   {
       root = q > nought ? -r/q : -sqrt(-q) ;
   }
   else if ( r > doubmax ||  r <  -doubmax ) //  x**3 = -r 
   {
       root =  -curoot(r) ;
   }
   else
*/
   {
       po3 = p*inv3 ;
       po3sq = po3*po3 ;

/*
       if (po3sq > doubmax) 
       {
           root =  -p ;
       }
       else
*/
       {
           v = r + po3*(po3sq + po3sq - q) ;

/*
           if ((v > doubmax) || (v < -doubmax)) 
           {
               root = -p ;
           }
           else
*/
           {
               vsq = v*v ;
               qo3 = q*inv3 ;
               uo3 = qo3 - po3sq ;
               u2o3 = uo3 + uo3 ;

/*
               if ((u2o3 > doubmax) || (u2o3 < -doubmax))
               {
                   root = p == nought ? ( q > nought ? -r/q : -sqrt(-q)  ) : -q/p ; 
               }
*/
              
               uo3sq4 = u2o3*u2o3 ;

/*
               if (uo3sq4 > doubmax)
               {
                   root = p == nought ? ( q > nought ? -r/q : -sqrt(fabs(q))  ) : -q/p ; 
               }
*/

               uo3cu4 = uo3sq4*uo3 ;
               wsq = uo3cu4 + vsq ;
               if (wsq >= nought)
               {
                   nrts = 1;  // cubic has one real root 
                   mcube = v <= nought ? ( -v + sqrt(wsq))*inv2 : ( -v - sqrt(wsq))*inv2 ;

                   m = cbrtf(mcube) ;
                   n = m != nought ? -uo3/m : nought ;

                   root = m + n - po3 ;
               }
               else
               {
                   nrts = 3;  // cubic has three real roots 

                   if (uo3 < nought)
                   {
                       muo3 = -uo3;
                       s = sqrt(muo3) ;
                       scube = s*muo3;
                       t =  -v/(scube+scube) ;
                       
                       //cosk = acos3(t) ;
                       cosk = cos(acos(t)*inv3) ;

                       if (po3 < nought)
                       {
                           root = (s+s)*cosk - po3;
                       }
                       else
                       {
                           sinsqk = doub1 - cosk*cosk ;
                           if (sinsqk < nought) sinsqk = nought ;
                           root = s*( -cosk - rt3*sqrt(sinsqk)) - po3 ;
                       }
                   }
                   else
                   {
                       root = cbrtf(v) - po3 ;
                   }
               }
           }
       }
   }
   return root ;
} 



