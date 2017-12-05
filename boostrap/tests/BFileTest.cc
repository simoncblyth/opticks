
#include <vector>
#include <string>
#include <cassert>


#include "BOpticksResource.hh"
#include "BFile.hh"

#include "SYSRAP_LOG.hh"
#include "BRAP_LOG.hh"
#include "PLOG.hh"


void test_FindFile_(const char* dirlist, const char* sub, const char* name)
{
    std::string aa = BFile::FindFile( dirlist, sub, name );
    if(!aa.empty())
    {
        printf("found sub %s name %s at %s \n", sub, name, aa.c_str());  
    }
    else
    {
        printf("NOT found sub %s name %s \n", sub, name);  
    }
} 

void test_FindFile()
{
   const char* dirlist = "$HOME/.opticks;$OPTICKS_PREFIX/.opticks" ;
   test_FindFile_( dirlist, "OpticksResource", "OpticksColors.json");
}


void test_ExistsDir()
{

    std::vector<std::string> ss ; 
    ss.push_back("$OPTICKS_HOME/optickscore/OpticksPhoton.h");
    ss.push_back("$HOME/.opticks/GColors.json");
    ss.push_back("$HOME/.opticks");
    ss.push_back("$HOME/");
    ss.push_back("$HOME");
    ss.push_back("$OPTICKS_HOME");
    ss.push_back("$HOME/$OPTICKS_HOME");
    ss.push_back("$TMP");

    for(unsigned int i=0 ; i < ss.size() ; i++)
    {
       std::string s = ss[i] ;
       std::string x = BFile::FormPath(s.c_str());

       bool xdir = BFile::ExistsDir(s.c_str());
       bool xfile = BFile::ExistsFile(s.c_str());

       bool xdir2 = BFile::ExistsNativeDir(x);
       bool xfile2 = BFile::ExistsNativeFile(x);

       assert( xdir == xdir2 );
       assert( xfile == xfile2 );


       printf("  BFile::FormPath(\"%s\") -->  [%s] dir %d file %d  \n", s.c_str(), x.c_str(), xdir, xfile);
    }
}


void test_CreateDir()
{
   BFile::CreateDir("$TMP/a/b/c");
}

void test_RemoveDir()
{
   BFile::CreateDir("$TMP/a/b/c");
   BFile::RemoveDir("$TMP/a/b/c");
}

void test_RemoveDir_2()
{
   BFile::CreateDir("$TMP","b","c");
   BFile::RemoveDir("$TMP","b","c");
}




void test_ParentDir()
{
    std::vector<std::string> ss ; 
    ss.push_back("$OPTICKS_HOME/optickscore/OpticksPhoton.h");
    ss.push_back("$HOME/.opticks/GColors.json");
    ss.push_back("C:\\tmp");
    ss.push_back("C:\\tmp\\TestIDPath");
 
    for(unsigned int i=0 ; i < ss.size() ; i++)
    {
       std::string s = ss[i] ;
       std::string x = BFile::FormPath(s.c_str());


       std::string p = BFile::ParentDir(s.c_str());

       LOG(info) 
               << " s " << std::setw(40) << s  
               << " x " << std::setw(40) << x  
               << " p " << std::setw(40) << p
               ;  

    } 

}



void test_FormPath_reldir()
{
    std::string x = BFile::FormPath("$TMP", "some/deep/reldir", "name.txt");

    LOG(info) << "test_FormPath_reldir"
              << " " << x 
              ;


}


void test_FormPath()
{
    std::vector<std::string> ss ; 
    ss.push_back("$OPTICKS_HOME/optickscore/OpticksPhoton.h");
    ss.push_back("$OPTICKS_INSTALL_PREFIX/include/optickscore/OpticksPhoton.h");
    ss.push_back("$OPTICKS_INSTALL_PREFIX/externals/config/geant4.ini") ;
    ss.push_back("$OPTICKS_INSTALL_PREFIX/opticksdata/config/opticksdata.ini") ;


    ss.push_back("$HOME/.opticks/GColors.json");
 
    for(unsigned int i=0 ; i < ss.size() ; i++)
    {
       std::string s = ss[i] ;
       std::string x = BFile::FormPath(s.c_str());

       LOG(info) 
               << " s " << std::setw(40) << s  
               << " x " << std::setw(40) << x  
               ;  
    }
}




void test_Name_ParentDir()
{
    const char* path = "$TMP/opticks/blyth/somefile.txt" ; 

    std::string name = BFile::Name(path) ;
    std::string dir = BFile::ParentDir(path) ;


    LOG(info) << " path " << path
              << " name " << name
              << " dir " << dir
              ;
 
}


void test_ChangeExt()
{
    const char* path = "$TMP/somedir/somefile.txt" ; 
    std::string name = BFile::Name(path) ;
    std::string stem = BFile::Stem(path);
    std::string dir = BFile::ParentDir(path) ;

    std::string chg = BFile::ChangeExt(path, ".json");


    LOG(info) << " path " << path
              << " name " << name
              << " stem " << stem
              << " dir " << dir
              << " chg " << chg
              ;
 

    
}

void test_SomeDir()
{
    //const char* path = "$TMP/somedir/someotherdir" ; 
    //const char* path = "/dd/Geometry/PoolDetails/lvVertiCableTray#pvVertiCable0xbf5e7f0" ;
    const char* path = "/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf" ; 

    std::string name = BFile::Name(path) ;
    std::string stem = BFile::Stem(path);
    std::string dir = BFile::ParentDir(path) ;


    LOG(info) 
              << "test_SomeDir"
              << " path " << path
              << " name " << name
              << " stem " << stem
              << " dir " << dir
              ;
 
}

void test_SomePath()
{
    //const char* path = "$TMP/somedir/someotherdir" ; 
    //const char* path = "/dd/Geometry/PoolDetails/lvVertiCableTray#pvVertiCable0xbf5e7f0" ;
    const char* path = "/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf" ; 

    std::string name = BFile::Name(path) ;
    std::string stem = BFile::Stem(path);
    std::string dir = BFile::ParentDir(path) ;

    LOG(info) 
              << "test_SomePath"
              << " path " << path
              << " name " << name
              << " stem " << stem
              << " dir " << dir
              ;
 
}





void test_LastWriteTime()
{
    LOG(info) << "test_LastWriteTime" ; 

    const char* path = "$TMP/a/b/c" ;
    std::time_t* lwt = BFile::LastWriteTime(path);
    std::time_t  now = std::time(NULL) ;


    if(!lwt)
    {
        std::cout 
          << " path " << path 
          << " DOESNT EXIST "
          << std::endl ; 
    }
    else
    {
        std::time_t age = (now - *lwt);
        std::cout 
          << " path " << path 
          << " now " << now
          << " age (s) " << age
          << " BFile::LastWriteTime(path) " << *lwt
          << std::endl 
          ; 
    }
}


void test_SinceLastWriteTime()
{
    LOG(info) << "test_SinceLastWriteTime" ; 
    //const char* path = "$TMP/a/b/c" ;
    const char* path = "$TMP/a/b" ;
    std::time_t* age = BFile::SinceLastWriteTime(path) ;
    if(age)
    {
        std::cout 
          << " path " << path 
          << " age : BFile::SinceLastWriteTime(path) " << *age
          << std::endl 
          ; 
    }
}



void test_LooksLikePath()
{
    LOG(info) << "test_LooksLikePath" ; 

    assert( BFile::LooksLikePath("$TMP/a/b") == true );
    assert( BFile::LooksLikePath("/a/b") == true );
    assert( BFile::LooksLikePath("1,2") == false );
    assert( BFile::LooksLikePath(NULL) == false );
    assert( BFile::LooksLikePath("1") == false );
}


void test_ParentName(const char* path, const char* expect)
{
    std::string pn = BFile::ParentName(path);

    LOG(info) << "test_ParentName"
              << " path [" << path << "]" 
              << " pn [" << pn << "]" 
              << " expect [" << expect << "]" 
              ; 

    if( expect == NULL )
    {
        assert( pn.empty() );
    }
    else
    {
        assert( pn.compare(expect) == 0 );
    }
}


void test_ParentName()
{
    test_ParentName( "/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae", "DayaBay_VGDX_20140414-1300" );
    test_ParentName( "DayaBay_VGDX_20140414-1300/g4_00.dae", "DayaBay_VGDX_20140414-1300" );
    test_ParentName( "g4_00.dae", NULL );
    test_ParentName( NULL, NULL );
}


void test_SplitPath(const char* path)
{
    std::vector<std::string> elem ; 
    BFile::SplitPath(elem, path);

    LOG(info) << " path " << path 
              << " nelem " << elem.size()
              ;

    for(unsigned i=0 ; i < elem.size() ; i++)
    {
        std::cout 
             << std::setw(4) << i 
             << " " << elem[i]
             << std::endl 
             ; 
    }
}


void test_SplitPath()
{
    const char* idpath_0="/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae" ; 
    test_SplitPath(idpath_0);

    //const char* idpath_1="/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1" ;  
    //test_SplitPath(idpath_1);
}






void test_prefixShorten_1()
{
    std::string path_ = BFile::FormPath("/some/other/dir/cfg4/DsG4OpBoundaryProcess.cc" ); 
    const char* path = path_.c_str();
    std::string abbr = BFile::prefixShorten(path, "$OPTICKS_HOME/" ); 
    LOG(info) 
             << " path [" << path << "]"
             << " abbr [" << abbr << "]"
             ; 

}


void test_prefixShorten_0()
{
    std::string path_ = BFile::FormPath("$OPTICKS_HOME/cfg4/DsG4OpBoundaryProcess.cc" ); 
    const char* path = path_.c_str();
    std::string abbr = BFile::prefixShorten(path, "$OPTICKS_HOME/" ); 
    LOG(info) 
             << " path [" << path << "]"
             << " abbr [" << abbr << "]"
             ; 

}







int main(int argc, char** argv)
{
   PLOG_(argc, argv);
   SYSRAP_LOG_ ;
   BRAP_LOG_ ;

   //BOpticksResource rsc ;  // sets envvar OPTICKS_INSTALL_PREFIX internally 
   //rsc.Summary();

   //test_FindFile();
   //test_ExistsDir();
   //test_CreateDir();
   //test_ParentDir();
   //test_FormPath();
   //test_Name_ParentDir();
   //test_ChangeExt();

   //test_FormPath_reldir();
   //test_SomeDir();
   //test_SomePath();
   //test_RemoveDir();
   //test_RemoveDir_2();


   //test_LastWriteTime();
   //test_SinceLastWriteTime();
   //test_LooksLikePath();
   //test_ParentName();
   //test_SplitPath();

   test_prefixShorten_0();
   test_prefixShorten_1();

   return 0 ; 
}



