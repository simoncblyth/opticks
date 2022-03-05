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

// om-;TEST=SPathTest om-t 

#include <cassert>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>

#include "SPath.hh"

#include "OPTICKS_LOG.hh"


void test_Stem()
{
    LOG(info); 
    const char* name = "hello.cu" ; 
    const char* stem = SPath::Stem(name); 
    const char* x_stem = "hello" ; 
    assert( strcmp( stem, x_stem ) == 0 ); 
}

void test_GetHomePath()
{
    LOG(info); 
    const char* bashrc = SPath::GetHomePath(".bashrc") ; 
    std::cout << bashrc << std::endl ; 
}

void test_IsReadable()
{
    LOG(info); 
    const char* self = SPath::GetHomePath("opticks/sysrap/tests/SPathTest.cc") ; 
    const char* non = SPath::GetHomePath("opticks/sysrap/tests/SPathTest.cc.non") ; 
    std::cout << self << std::endl ; 
    bool readable = SPath::IsReadable(self); 
    if(!readable)
    {
       LOG(fatal) << "looks like opticks source not in HOME" ;  
    }
    //assert( readable == true ); 

    bool readable_non = SPath::IsReadable(non); 
    assert( readable_non == false ); 
}

void test_Dirname()
{
    LOG(info); 
    const char* lines = R"LIT(
$HOME/hello 
$TMP/somewhere/over/the/rainbow.txt
$NON_EXISTING_EVAR/elsewhere/sub.txt
/just/some/path.txt
stem.ext
/
$
)LIT";
    std::stringstream ss(lines); 
    std::string line ;
    while (std::getline(ss, line, '\n'))
    {
        if(line.empty()) continue ; 
        const char* path = SPath::Dirname(line.c_str()); 
        std::cout 
            << std::setw(60) << line
            << " : "
            << std::setw(60) << path
            << std::endl 
            ; 
    }
}


void test_Basename()
{
    LOG(info); 
    std::vector<std::string> paths = { "/dd/materials/Water", "Water", "" } ; 
    for(unsigned i=0 ; i < paths.size() ; i++)
    {
        const char* path = paths[i].c_str() ;
        const char* base = SPath::Basename(path) ;
        std::cout 
            << " path [" << path << "]"  
            << " base [" << base << "]" 
            << std::endl 
            ;  
    }
}

void test_UserTmpDir()
{
    LOG(info); 
    const char* tmp = SPath::UserTmpDir(); 
    std::cout << tmp << std::endl ; 
}

void test_Resolve()
{
    LOG(info); 
    const char* lines = R"LIT(
$TMP
$OPTICKS_TMP
$OPTICKS_EVENT_BASE
$HOME/hello 
$TMP/somewhere/over/the/rainbow.txt
$NON_EXISTING_EVAR/elsewhere/sub.txt
/just/some/path.txt
stem.ext
/
$
)LIT";
    std::stringstream ss(lines); 
    std::string line ;
    while (std::getline(ss, line, '\n'))
    {
        if(line.empty()) continue ; 
        int create_dirs = 0 ; // noop
        const char* path = SPath::Resolve(line.c_str(), create_dirs); 
        std::cout 
            << std::setw(60) << line
            << " : "
            << std::setw(60) << path
            << std::endl 
            ; 
    }
}


void test_ChangeName()
{
    const char* srcpath = "/some/long/path/ending/with/pixels.jpg" ;  
    const char* path = SPath::ChangeName(srcpath, "posi.npy" ); 
    const char* xpath = "/some/long/path/ending/with/posi.npy" ;  
    LOG(info) << path ; 
    assert( 0 == strcmp( path, xpath ) ) ;
}

void test_MakeDirs()
{
    const char* path = "/tmp/SPathTest/red/green/blue" ; 
    int rc = SPath::MakeDirs(path); 
    LOG(info) << " path " << path << " rc " << rc ; 
}


void test_MakePath()
{
    const char* path = SPath::MakePath<double>("/tmp/SPathTest", "test_MakePath", 1.123, "demo.npy" );  
    LOG(info) << " path " << path  ; 
}

void test_Resolve_createdirs()
{
    const char* path = "$TMP/red/green/blue/file.txt" ; 
    int create_dirs = 1 ; // 1:filepath 
    const char* p = SPath::Resolve(path, create_dirs); 

    LOG(info) << path << " " << p ; 

    std::ofstream fp(p, std::ios::out); 
    fp << path ; 
    fp.close(); 
}

void test_getcwd()
{
    const char* cwd = SPath::getcwd() ; 
    LOG(info) << " after SPath::chdir SPath::getcwd " << cwd  ; 
} 

void test_mtime()
{
    const char* path = "/tmp/tt.txt" ; 
    int mtime = SPath::mtime(path);  
    LOG(info) << " path " << path << " mtime " << mtime ; 
}



int main(int argc , char** argv )
{
    SPath::chdir("$TMP/red/green/blue/logs");  
    // chdir before OPTICKS_LOG succeeds to write logfile named after executable into that directory 

    OPTICKS_LOG(argc, argv);

/*
    test_Stem();  
    test_GetHomePath();  
    test_IsReadable();  
    test_Dirname(); 
    test_Basename(); 
    test_UserTmpDir(); 
    test_Resolve(); 
    test_ChangeName(); 
    test_MakeDirs(); 
    test_MakePath(); 
    test_Resolve_createdirs(); 
    test_getcwd(); 
*/
    test_mtime(); 

    return 0  ; 
}
// om-;TEST=SPathTest om-t 
