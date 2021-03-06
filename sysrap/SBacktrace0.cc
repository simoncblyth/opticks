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



void printStackTrace_0()
{
    void *returnAddresses[500];
    int depth = backtrace(returnAddresses, sizeof returnAddresses / sizeof *returnAddresses);
    printf("stack depth = %d\n", depth);
    char **symbols = backtrace_symbols(returnAddresses, depth);
    for (int i = 0; i < depth; ++i) {
        printf("%s\n", symbols[i]);
    }
    free(symbols);
}



 
static inline void printStackTrace( FILE *out = stderr, unsigned int max_frames = 63 )
{
   fprintf(out, "stack trace:\n");
 
   // storage array for stack trace address data
   void* addrlist[max_frames+1];
 
   // retrieve current stack addresses
   unsigned int addrlen = backtrace( addrlist, sizeof( addrlist ) / sizeof( void* ));
 
   if ( addrlen == 0 ) 
   {
      fprintf( out, "  \n" );
      return;
   }
 
   // resolve addresses into strings containing "filename(function+address)",
   // Actually it will be ## program address function + offset
   // this array must be free()-ed
   char** symbollist = backtrace_symbols( addrlist, addrlen );

   for ( unsigned i = 0 ; i < addrlen; i++ )
   {
       fprintf( out, "%s\n", symbollist[i]);
   }
 
   size_t funcnamesize = 1024;
   char funcname[1024];

      // find parentheses and +address offset surrounding the mangled name
#ifdef __APPLE__
   for ( unsigned int i = 4; i < addrlen; i++ )
   {
      char* begin_name   = NULL;
      char* begin_offset = NULL;

      for ( char *p = symbollist[i]; *p; ++p )
      {
         if (( *p == '_' ) && ( *(p-1) == ' ' )) begin_name = p-1;
         else if ( *p == '+' ) begin_offset = p-1;
      }
 
      if ( begin_name && begin_offset && ( begin_name < begin_offset ))
      {
         *begin_name++ = '\0';
         *begin_offset++ = '\0';
 
         // mangled name is now in [begin_name, begin_offset) and caller
         // offset in [begin_offset, end_offset). now apply
         // __cxa_demangle():
         int status;
         char* ret = abi::__cxa_demangle( begin_name, &funcname[0],
                                          &funcnamesize, &status );
         if ( status == 0 ) 
         {
            //funcname = ret; // use possibly realloc()-ed string
            strncpy( funcname, ret, funcnamesize  );

            fprintf( out, " : %-30s : %-40s : %s :\n",
                     symbollist[i], funcname, begin_offset );
         } else {
            // demangling failed. Output function name as a C function with
            // no arguments.
            fprintf( out, " : %-30s %-38s() %s\n",
                     symbollist[i], begin_name, begin_offset );
         }

      } else {
         // couldn't parse the line? print the whole line.
         fprintf(out, "  %-40s\n", symbollist[i]);
      }
   }
 
#else 

   for ( unsigned int i = 3 ; i < addrlen; i++ )
   {
      char* begin_name   = NULL;
      char* begin_offset = NULL;
      char* end_offset   = NULL;
      // ./module(function+0x15c) [0x8048a6d]
      for ( char *p = symbollist[i]; *p; ++p )
      {
         if ( *p == '(' )
            begin_name = p;
         else if ( *p == '+' )
            begin_offset = p;
         else if ( *p == ')' && ( begin_offset || begin_name ))
            end_offset = p;
      }
 
      if ( begin_name && end_offset && ( begin_name < end_offset ))
      {
         *begin_name++   = '\0';
         *end_offset++   = '\0';
         if ( begin_offset )
            *begin_offset++ = '\0';
 
         // mangled name is now in [begin_name, begin_offset) and caller
         // offset in [begin_offset, end_offset). now apply
         // __cxa_demangle():
 
         int status = 0;
         char* ret = abi::__cxa_demangle( begin_name, funcname,
                                          &funcnamesize, &status );
         char* fname = begin_name;
         if ( status == 0 ) 
            fname = ret;
 
         if ( begin_offset )
         {
            fprintf( out, "  %-30s ( %-40s  + %-6s) %s\n",
                     symbollist[i], fname, begin_offset, end_offset );
         } else {
            fprintf( out, "  %-30s ( %-40s    %-6s) %s\n",
                     symbollist[i], fname, "", end_offset );
         }


      } else {
         // couldn't parse the line? print the whole line.
         fprintf(out, "  %-40s\n", symbollist[i]);
      }
   }
 
#endif  // !DARWIN - but is posix

   free(symbollist);
}


