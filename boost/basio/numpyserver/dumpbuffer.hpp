#pragma once
#include "stdio.h"

inline void DumpBuffer(const char* buffer, size_t buflen, size_t maxlines ) 
{
   const char* hfmt = "  %s \n%06X : " ;

   int ascii[2] = { 0x20 , 0x7E };
   const int N = 16 ;
   size_t halfmaxbytes = N*maxlines/2 ; 

   char line[N+1] ;
   int n = N ; 
   line[n] = '\0' ;
   while(n--) line[n] = ' ' ;

   for (size_t i = 0; i < buflen ; i++){
       int v = buffer[i] & 0xff ;
       bool out = i < halfmaxbytes || i > buflen - halfmaxbytes - 1 ; 
       if( i == halfmaxbytes || i == buflen - halfmaxbytes - 1  ) printf(hfmt, "...", i );  
       if(!out) continue ; 

       int j = i % N ; 
       if(j == 0) printf(hfmt, line, i );  // output the prior line and start new one with byte counter  
       line[j] = ( v >= ascii[0] && v < ascii[1] ) ? v : '.' ;  // ascii rep 
       printf("%02X ", v );
   }   
   printf(hfmt, line, buflen );
   printf("\n"); 
}


