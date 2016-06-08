#include "fsutil.hh"
#include "dbg.hh"

#include <vector>
#include <string>
#include <cassert>


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




/*
   DBG(argv[0]," fsutil::FormPath(\"/tmp\",\"name.npy\") ", fsutil::FormPath("/tmp","name.npy") ); 
   DBG(argv[0]," fsutil::FormPath(\"/tmp\",\"sub\",\"name.npy\") ", fsutil::FormPath("/tmp","sub","name.npy") ); 


   fsutil::CreateDir("/tmp/a/b/c");
*/

   return 0 ; 
}



