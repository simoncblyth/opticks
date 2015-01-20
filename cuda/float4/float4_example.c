/*
   clang -I$(cuda-dir)/include float4_example.c && ./a.out

*/


#include "vector_types.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h> 


void dump( float4* a , unsigned int N)
{
   for(int i=0 ; i< N ;++i ) printf("a[%d]  %f %f %f %f  \n", i, a[i].x,a[i].y,a[i].z,a[i].w );
}

void fill( float4* a, unsigned int N)
{

   for(int i=0 ; i< N ;++i )
   {
      float f = (float)i ; 
      a[i].x = f+0.1 ; 
      a[i].y = f+0.2 ; 
      a[i].z = f+0.3 ; 
      a[i].w = f+0.4 ; 
   } 
}

int main()
{
   unsigned int N = 10 ;
 
   float4* a = (float4 *)malloc(sizeof(float4)*N);   

   fill(a, N);
   dump(a, N); 
   printf("---\n");

   memcpy( a+5, a+1 , sizeof(float4)*3 );
   //memcpy( (void*)&a[5], (const void*)&a[1] , sizeof(float4)*3 );

   dump(a, N); 

   return 0 ; 
}



