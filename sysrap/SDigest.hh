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
SDigest
========

MD5 based hexdigest machinery, used throughout Opticks.
Allows incremental update building of the digest.


DevNotes
----------
   
* http://stackoverflow.com/questions/7627723/how-to-create-a-md5-hash-of-a-string-in-c
* hails from env/base/hash/md5digest.h

**/


#include <string>
#include <vector>
#include "plog/Severity.h"


#if defined __APPLE__

#   define COMMON_DIGEST_FOR_OPENSSL
#   include <CommonCrypto/CommonDigest.h>
#   define SHA1 CC_SHA1

#elif defined _MSC_VER

#   include "md5.hh"

#elif __linux

#   include <openssl/md5.h>

#endif

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SDigest 
{
       static const plog::Severity LEVEL ;    
   public:
       static const char* hexchar ; 
       static bool IsDigest(const char* s);

       static std::string DigestPathInByteRange(const char* path, int i0, int i1, unsigned bufsize=8192); 
       static std::string DigestPath(const char* path, unsigned bufsize=8192);
       static std::string DigestPath2(const char* path);

       static const char* md5digest_( const char* buffer, int len );
       static std::string md5digest( const char* buffer, int len );
       static std::string digest( void* buffer, int len );
       static std::string digest( std::vector<std::string>& ss);
       static std::string digest_skipdupe( std::vector<std::string>& ss);
       

   public:
       SDigest();
       virtual ~SDigest();
   public:
       void update( char* buffer, int length);
       void update_str( const char* str );
       void update( const std::string& str );
       char* finalize();
   private:
       MD5_CTX m_ctx ;

};




