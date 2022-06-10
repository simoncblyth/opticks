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

#include <string>
#include <cstring>
#include <cassert>
#include <sstream>
#include <fstream>
#include <iostream>


#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <errno.h>

// Linux specific, but works on Darwin via some compat presumably  
#include <sys/stat.h>
#include <unistd.h>    // for chdir


#include "SStr.hh"
#include "SPath.hh"
#include "SOpticksResource.hh"
#include "PLOG.hh"


const plog::Severity SPath::LEVEL = PLOG::EnvLevel("SPath", "DEBUG"); 


const char* SPath::Stem( const char* name ) // static
{
    std::string arg = name ;
    std::string base = arg.substr(0, arg.find_last_of(".")) ; 
    return strdup( base.c_str() ) ; 
}

bool SPath::IsReadable(const char* base, const char* name)  // static 
{
    std::stringstream ss ; 
    ss << base << "/" << name ; 
    std::string s = ss.str(); 
    return IsReadable(s.c_str()); 
}
bool SPath::IsReadable(const char* path)  // static 
{
    std::ifstream fp(path, std::ios::in|std::ios::binary);
    bool readable = !fp.fail(); 
    fp.close(); 
    return readable ; 
}

const char* SPath::GetHomePath(const char* rel)  // static 
{
    char* home = getenv("HOME"); 
    assert(home);  
    std::stringstream ss ; 
    ss << home ;
    if(rel != NULL) ss << "/" << rel ; 
    std::string path = ss.str(); 
    return strdup(path.c_str()) ; 
}

/**
SPath::Dirname
-----------------

::

   SPath::Dirname("/some/path/with/some/last/elem") ->  "/some/path/with/some/last"

**/

const char* SPath::Dirname(const char* path)  
{
    std::string p = path ; 
    std::size_t pos = p.find_last_of("/");
    std::string dir = pos == std::string::npos ? p : p.substr(0,pos) ; 
    return strdup( dir.c_str() ) ; 
}


const char* SPath::ChangeName(const char* srcpath, const char* name)
{
    const char* dir = Dirname(srcpath); 
    std::stringstream ss ; 
    ss << dir << "/" << name ; 
    free((void*)dir) ; 
    std::string path = ss.str(); 
    return strdup( path.c_str() );   
}


const char* SPath::Basename(const char* path)
{
    std::string p = path ; 
    std::size_t pos = p.find_last_of("/");
    std::string base = pos == std::string::npos ? p : p.substr(pos+1) ; 
    return strdup( base.c_str() ) ; 
}

const char* SPath::UserTmpDir(const char* pfx, const char* user_envvar, const char* sub, char sep  ) // static 
{
    char* user = getenv(user_envvar); 
    std::stringstream ss ; 
    ss << pfx 
       << sep
       << user 
       << sep
       << sub 
       ; 
    std::string s = ss.str(); 
    return strdup(s.c_str()); 
}


/**
SPath::Resolve
---------------

Resolves tokenized paths such as "$PREFIX/name.ext" where PREFIX must 
be an existing envvar. Special handling if "$TMP" is provided that defaults 
the TMP envvar to "/tmp/username/opticks" 

HUH: the special handling like it only works for "$TMP" not eg for "$TMP/some/thing.txt"

Hmm need to distinguish when the path is a folder or a file for create_dirs ?

**/

const char* SPath::Resolve(const char* spec_, int create_dirs)
{
    LOG(LEVEL) 
        << " spec_ [" << spec_ << "]"
        << " create_dirs [" << create_dirs << "]"
        ;

    if(!spec_) return nullptr ;       

    char* spec = strdup(spec_);            // copy to allow modifications 
    char sep = '/' ; 
    char* spec_sep = strchr(spec, sep);    // pointer to first separator : so its the string starting from the first sep 
    char* spec_end = strchr(spec, '\0') ;  // pointer to null terminator

    LOG(LEVEL) 
        << " spec " << spec 
        << " spec_sep " << spec_sep
        << " spec_end " << spec_end
        ;

    bool tok_rel = spec[0] == '$' && spec_sep && spec_end && spec_sep != spec_end ; 
    bool tok_only = spec[0] == '$' && spec_sep == nullptr ; 

    std::stringstream ss ; 
    if(tok_rel)
    {
        *spec_sep = '\0' ;                 // temporarily null terminate at the first slash  
        const char* pfx = SOpticksResource::Get(spec+1);      // start one past the dollar terminated at first slash  
        *spec_sep = sep ;                  // put back the separator
        const char* prefix = pfx ? pfx : UserTmpDir() ; 
        ss << prefix << spec_sep ; 
    }
    else if(tok_only)
    {
        const char* pfx = SOpticksResource::Get(spec+1); 
        const char* prefix = pfx ? pfx : UserTmpDir() ;
        ss << prefix ; 
    }
    else
    {
        ss << spec ; 
    }
    std::string s = ss.str(); 

    const char* path = s.c_str(); 

    CreateDirs(path, create_dirs ); 

    return strdup(path) ; 
}


/**
SPath::CreateDirs
--------------------

mode:0 
    do nothing 

mode:1
    create directories assuming the path argumnent is a file path by using SPath::Dirname 
    to extract the directory

mode:2
    create directories assuming the path argumnent is a directory path 

**/

void SPath::CreateDirs(const char* path, int mode)  // static 
{
    if(mode == 0) return ; 
    const char* dirname = mode == 1 ? Dirname(path) : path ; 
    int rc = MakeDirs(dirname) ; 
    LOG(LEVEL)
        << " path " << path 
        << " mode " << mode
        << " dirname " << dirname
        << " rc " << rc
        ;
    assert( rc == 0 ); 
}



const char* SPath::Resolve(const char* dir, const char* name, int create_dirs)
{
    LOG(LEVEL) 
        << " dir [" << dir << "]"
        << " name [" << ( name  ? name : "-" )<< "]"
        << " create_dirs [" << create_dirs << "]"
        ;

    std::stringstream ss ; 
    ss << dir ; 
    if(name) ss << "/" << name ; 

    std::string s = ss.str(); 
    return Resolve(s.c_str(), create_dirs); 
}

const char* SPath::Resolve(const char* dir, const char* reldir, const char* name, int create_dirs)
{
    LOG(LEVEL) 
        << " dir [" << dir << "]"
        << " reldir [" << reldir << "]"
        << " name [" << name << "]"
        << " create_dirs [" << create_dirs << "]"
        ;

    std::stringstream ss ; 
    if(dir)    ss << dir << "/" ; 
    if(reldir) ss << reldir << "/" ; 
    ss << name ; 

    std::string s = ss.str(); 
    return Resolve(s.c_str(), create_dirs); 
}



const char* SPath::Resolve(const char* dir, const char* reldir, const char* rel2dir, const char* name, int create_dirs)
{
    LOG(LEVEL) 
        << " dir [" << dir << "]"
        << " reldir [" << reldir << "]"
        << " rel2dir [" << rel2dir << "]"
        << " name [" << name << "]"
        << " create_dirs [" << create_dirs << "]"
        ;

    std::stringstream ss ; 
    if(dir)    ss << dir << "/" ; 
    if(reldir) ss << reldir << "/" ; 
    if(rel2dir) ss << rel2dir << "/" ; 
    if(name) ss << name ; 

    std::string s = ss.str(); 
    return Resolve(s.c_str(), create_dirs); 
}



const char* SPath::Resolve(const char* dir, const char* reldir, const char* rel2dir, const char* rel3dir, const char* name, int create_dirs)
{
    LOG(LEVEL) 
        << " dir [" << dir << "]"
        << " reldir [" << reldir << "]"
        << " rel2dir [" << rel2dir << "]"
        << " rel3dir [" << rel3dir << "]"
        << " name [" << name << "]"
        << " create_dirs [" << create_dirs << "]"
        ;

    std::stringstream ss ; 
    ss << dir << "/" ; 
    if(reldir) ss << reldir << "/" ; 
    if(rel2dir) ss << rel2dir << "/" ; 
    if(rel3dir) ss << rel3dir << "/" ; 
    if(name) ss << name ; 

    std::string s = ss.str(); 
    return Resolve(s.c_str(), create_dirs); 
}



const char* SPath::Resolve(const char* dir, int idx, int create_dirs)
{
    return SPath::Resolve(dir, SStr::FormatIndex(idx), create_dirs); 
}
const char* SPath::Resolve(const char* dir, const char* name, int idx, int create_dirs)
{
    return SPath::Resolve(dir, name, SStr::FormatIndex(idx), create_dirs); 
}
const char* SPath::Resolve(const char* dir, const char* reldir, const char* name, int idx, int create_dirs)
{
    return SPath::Resolve(dir, reldir, name, SStr::FormatIndex(idx), create_dirs); 
}










bool SPath::LooksLikePath(const char* path)
{
    if(!path) return false ;
    if(strlen(path) < 2) return false ; 
    return path[0] == '/' || path[0] == '$' ; 
}

/**
SPath::MakeDirs
----------------

See sysrap/tests/mkdirp.cc

**/

int SPath::MakeDirs( const char* path_, int mode_ )
{
    mode_t default_mode = S_IRWXU | S_IRGRP |  S_IXGRP | S_IROTH | S_IXOTH ;
    mode_t mode = mode_ == 0 ? default_mode : mode_ ;  

    char* path = strdup(path_);
    char* p = path ;  
    int rc = 0 ;  

    while (*p != '\0' && rc == 0)
    {   
        p++;                                 // advance past leading character, probably slash, and subsequent slashes the next line gets to  
        while(*p != '\0' && *p != '/') p++;  // advance p until subsequent slash 
        char v = *p;                         // store the slash      
        *p = '\0' ;                          // replace slash with string terminator
        //printf("%s\n", path );                   
        rc = mkdir(path, mode) == -1 && errno != EEXIST ? 1 : 0 ;  // set rc non-zero for mkdir errors other than exists already  
        *p = v;                              // put back the slash  
    }       
    free(path);
    return rc ;
}

void SPath::chdir(const char* path, int create_dirs)  // static
{
    assert( create_dirs == 0 || create_dirs == 2 );   // 0:do nothing OR 2:dirpath

    const char* p = SPath::Resolve(path, create_dirs);  

    std::cout << "SPath::chdir " << p << std::endl ; 

    int rc = ::chdir(p) ; 

    assert( rc == 0 ); 
}

const char* SPath::getcwd()  // static
{
    char path[100] ; 
    char* ret = ::getcwd(path, 100); 
    return ret == nullptr ? nullptr : strdup(path); 
}

void SPath::MakeEmpty(const char* path_)
{
    const char* path = SPath::Resolve(path_, FILEPATH); 
    std::ofstream fp(path);
}

bool SPath::Exists(const char* path_) // static 
{
    const char* path = SPath::Resolve(path_, FILEPATH); 
    std::ifstream fp(path, std::ios::in|std::ios::binary);
    return fp.fail() ? false : true ; 
}

const char* SPath::PickFirstExisting(const char* path0, const char* path1, const char* path2 )
{
    if(SPath::Exists(path0)) return path0 ; 
    if(SPath::Exists(path1)) return path1 ; 
    if(SPath::Exists(path2)) return path2 ; 
    return nullptr ; 
}


const char* SPath::SomeGDMLPath()
{
    // TODO: use GDXML instead of the old CGDMLKludge 
    const char* path0 = getenv("SomeGDMLPath") ;   
    const char* path1 = SPath::Resolve("$IDPath/origin_CGDMLKludge.gdml", NOOP );  
    const char* path2 = SPath::Resolve("$OPTICKS_PREFIX/origin_CGDMLKludge_02mar2022.gdml", NOOP) ; 

    const char* path = SPath::PickFirstExisting(path0, path1, path2); 
    std::cout << " path " << ( path ? path : "-" ) << std::endl ;    
    assert(path); 
    return path ; 
}


int SPath::Remove(const char* path_)
{
    const char* path = SPath::Resolve(path_, FILEPATH); 
    assert( strlen(path) > 2 ); 
    return remove(path);  
}

int SPath::mtime(const char* path)
{
     struct stat st;
     int ierr = stat(path, &st ); 
     int mtime = st.st_mtime;
     return ierr == 0 ? mtime : 0 ; 
}



template<typename T>
const char* SPath::MakePath( const char* prefix, const char* reldir, const T real, const char* name)  // static
{
    const char* sreal = SStr::FormatReal<T>(real, 6, 4, '0');
    int create_dirs = 2 ;  // 2:dirpath
    const char* fold = SPath::Resolve(prefix, reldir, sreal, create_dirs ); 
    const char* path = SPath::Resolve(fold, name, 0 ) ;  // 0:create_dirs nop
    return path ; 
} 

template const char* SPath::MakePath<float>( const char*, const char*, const float, const char* ); 
template const char* SPath::MakePath<double>( const char*, const char*, const double, const char* ); 


/**
SPath::MakeName
------------------

Form a name from the arguments::

    <stem><index><ext>

stem
    can be null
index
    ignored when < 0, otherwise yields %0.5d formatted string 
ext 
    typically includes dot, eg ".jpg" or ".npy"


**/

std::string SPath::MakeName( const char* stem, int index, const char* ext )
{ 
    std::string name ; 
    if( index > -1 )
    {
        if( stem && ext )
        {
            name = SStr::Format("%s%0.5d%s", stem, index, ext ) ; 
        }
        else if( stem == nullptr && ext )
        {
            name = SStr::Format("%0.5d%s", index, ext ) ; 
        }
    }
    else
    {
        if( stem && ext )
        {
            std::stringstream ss ; 
            ss << stem << ext ; 
            name = ss.str(); 
        }
        else if( stem == nullptr && ext )
        {
            name = ext ; 
        }
    }
    return name ; 
}


/**
SPath::Make
-------------

Creates a path from the arguments::

    <base>/<reldir>/<stem><index><ext>

* base and relname can be nullptr 
* the stem index and ext are formatted using SPath::MakeName
* directory is created 

**/

const char* SPath::Make( const char* base, const char* reldir, const char* stem, int index, const char* ext, int create_dirs )
{
    assert( create_dirs == NOOP || create_dirs == FILEPATH ); 
    std::string name = MakeName(stem, index, ext); 
    const char* path = SPath::Resolve(base, reldir, name.c_str(), create_dirs ) ; 
    return path ; 
}

/**
SPath::Make
-------------

    <base>/<reldir>/<reldir2>/<stem><index><ext>

**/

const char* SPath::Make( const char* base, const char* reldir, const char* reldir2, const char* stem, int index, const char* ext, int create_dirs )
{
    assert( create_dirs == NOOP || create_dirs == FILEPATH ); 
    std::string name = MakeName(stem, index, ext); 
    const char* path = SPath::Resolve(base, reldir, reldir2, name.c_str(), create_dirs ) ; 
    return path ; 
}



