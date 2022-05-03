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
SPath
======

**/

#include <string>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

enum { NOOP, FILEPATH, DIRPATH } ;

class SYSRAP_API SPath {
  public:
      static const plog::Severity LEVEL ; 
      static const char* Stem( const char* name );
      static bool IsReadable(const char* base, const char* name);
      static bool IsReadable(const char* path);
      static const char* GetHomePath(const char* rel); 
      static const char* Dirname(const char* path); 
      static const char* Basename(const char* path); 
      static const char* ChangeName(const char* srcpath, const char* name) ;

      static int mtime(const char* path); 
      


      static const char* UserTmpDir(const char* pfx="/tmp", const char* user_envvar="USER", const char* sub="opticks", char sep='/'  );

      // create_dirs:(0 do nothing, 1:assume file path, 2:assume dir path)
      static const char* Resolve(const char* path, int create_dirs); 
      static const char* Resolve(const char* dir, const char* name, int create_dirs);
      static const char* Resolve(const char* dir, const char* reldir, const char* name, int create_dirs);
      static const char* Resolve(const char* dir, const char* reldir, const char* rel2dir, const char* name, int create_dirs);
      static const char* Resolve(const char* dir, const char* reldir, const char* rel2dir, const char* rel3dir, const char* name, int create_dirs);




      // mode:(0 do nothing, 1:assume file path, 2:assume dir path) 
      static void CreateDirs(const char* path, int mode); 

      static bool LooksLikePath(const char* path);
      static int MakeDirs( const char* path, int mode=0 ) ; 
      static void chdir(const char* path, int create_dirs=2 ); 
      static const char* getcwd() ; 

      template<typename T> static const char* MakePath( const char* prefix, const char* reldir, const T real, const char* name); 


      static std::string MakeName( const char* stem, int index, const char* ext ); 
      static const char* Make( const char* base, const char* reldir,                      const char* stem, int index, const char* ext, int create_dirs ); 
      static const char* Make( const char* base, const char* reldir, const char* reldir2, const char* stem, int index, const char* ext, int create_dirs ); 


};



