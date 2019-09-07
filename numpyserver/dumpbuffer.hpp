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


