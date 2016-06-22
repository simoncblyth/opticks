
#include <vector>
#include <string>
#include <cassert>


#include "BFile.hh"
#include "PLOG.hh"
#include "BRAP_LOG.hh"


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
    ss.push_back("$ENV_HOME/optickscore/OpticksPhoton.h");
    ss.push_back("$HOME/.opticks/GColors.json");
    ss.push_back("$HOME/.opticks");
    ss.push_back("$HOME/");
    ss.push_back("$HOME");
    ss.push_back("$ENV_HOME");
    ss.push_back("$HOME/$ENV_HOME");
    ss.push_back("/tmp");

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
   BFile::CreateDir("/tmp/a/b/c");
}

void test_ParentDir()
{
    std::vector<std::string> ss ; 
    ss.push_back("$ENV_HOME/optickscore/OpticksPhoton.h");
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





int main(int argc, char** argv)
{
   PLOG_(argc, argv);
   BRAP_LOG_ ;

   //test_FindFile();
   //test_ExistsDir();
   //test_CreateDir();
   test_ParentDir();



   return 0 ; 
}



