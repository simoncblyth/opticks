#pragma once

#include <cstddef>

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SSys {
  public:
     static void WaitForInput(const char* msg="Enter any key to continue...\n");
     static int getenvint( const char* envkey, int fallback=-1 );
     static int atoi_( const char* a );
     static const char* getenvvar( const char* envprefix, const char* envkey, const char* fallback=NULL );
     static const char* getenvvar( const char* envkey );
     static int setenvvar( const char* envprefix, const char* key, const char* value, bool overwrite=true );
     static bool IsRemoteSession();
     static bool IsCTestInteractiveDebugMode();

};
