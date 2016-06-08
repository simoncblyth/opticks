#include "fsutil.hh"
#include "dbg.hh"

#include <vector>
#include <string>
#include <cassert>





void test_find(const char* dirlist, const char* sub, const char* name)
{
    std::string aa = fsutil::FindFile( dirlist, sub, name );
    if(!aa.empty())
    {
        printf("found sub %s name %s at %s \n", sub, name, aa.c_str());  
    }
    else
    {
        printf("NOT found sub %s name %s \n", sub, name);  
    }
} 




int main(int argc, char** argv)
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
       std::string x = fsutil::FormPath(s.c_str());

       bool xdir = fsutil::ExistsDir(s.c_str());
       bool xfile = fsutil::ExistsFile(s.c_str());

       bool xdir2 = fsutil::ExistsNativeDir(x);
       bool xfile2 = fsutil::ExistsNativeFile(x);

       assert( xdir == xdir2 );
       assert( xfile == xfile2 );


       printf("  fsutil::FormPath(\"%s\") -->  [%s] dir %d file %d  \n", s.c_str(), x.c_str(), xdir, xfile);
    }



   const char* dirlist = "$HOME/.opticks;$OPTICKS_PREFIX/.opticks" ;
   test_find( dirlist, "OpticksResource", "OpticksColors.json");



   //fsutil::CreateDir("/tmp/a/b/c");

   return 0 ; 
}



