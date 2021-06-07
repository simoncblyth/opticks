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
/**
SSys
=====

static functions for system level things
such as accessing envvars, running external executables, 
and detecting the interactivity level of the session.


**/


#include <cstddef>
#include <string>
#include <vector>
#include "plog/Severity.h"

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SSys {
  public:

     static const plog::Severity LEVEL ; 
     static const unsigned SIGNBIT32 ;
     static const unsigned OTHERBIT32 ; 

     static bool IsNegativeZero(float f);
     static void DumpEnv(const char* pfx="OPTICKS" ); // static

     static const char* fmt(const char* tmpl="hello%u.npy", unsigned val=0);
     static int run(const char* cmd);
     static int exec(const char* exe, const char* arg1, const char* arg2=NULL);
     static std::string Which(const char* script); 
     static std::string POpen(const char* cmd, bool chomp, int& rc); 
     static std::string POpen(const char* cmda, const char* cmdb, bool chomp, int& rc); 


     static int npdump(const char* path="$TMP/torchstep.npy", const char* nptype="np.int32", const char* postview=NULL, const char* printoptions=NULL);
     static void xxdump(char* buf, int num_bytes, int width=16, char non_printable='.' ); 
     static std::string xxd(char* buf, int num_bytes, int width=16, char non_printable='.' ); 
     static std::string hexlify(const void* obj, size_t size, bool reverse=true ) ; 

     static void WaitForInput(const char* msg="Enter any key to continue...\n");
     static int  getenvint( const char* envkey, int fallback=-1 );
     static float getenvfloat( const char* envkey, float fallback=-1.f );
     static bool getenvbool( const char* envkey );
     static int getenvintvec( const char* envkey, std::vector<int>& ivec, char delim=',' );


     static int   atoi_( const char* a );
     static float atof_( const char* a );

     static void split(std::vector<std::string>& elem, const char* str, char delim );

     static const char* getenvvar( const char* envkey, const char* fallback );
     static const char* getenvvar( const char* envkey );
     static const char* username(); 
     static const char* hostname(); 

     static int setenvvar( const char* ekey, const char* value, bool overwrite=true, char special_empty_token='\0' );
     static bool IsRemoteSession();
     static bool IsVERBOSE();
     static bool IsHARIKARI();
     static bool IsENVVAR(const char* envvar);
     static int GetInteractivityLevel();
     static bool IsCTestRunning();

     static int OKConfCheck();


     static unsigned COUNT ; 
     static void Dump_(const char* msg);
     static void Dump(const char* msg);

     static const char* ResolveExecutable(const char* envvar_key, const char* default_executable);
     static const char* ResolvePython();
     static int RunPythonScript(const char* script);
     static int RunPythonCode(const char* code);


};
