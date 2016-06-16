#pragma once

#include <cstddef>


class BSys {
  public:
     static void WaitForInput(const char* msg="Enter any key to continue...\n");
     static int getenvint( const char* envkey, int fallback=-1 );
     static const char* getenvvar( const char* envprefix, const char* envkey, const char* fallback=NULL );
     static int setenvvar( const char* envprefix, const char* key, const char* value, bool overwrite=true );


};
