#include <iostream>
#include <string>
#include <cstring>
#include <boost/filesystem.hpp>
#include <boost/version.hpp>

#define API  __attribute__ ((visibility ("default")))

struct API UseBoostFS 
{
   static const char* concat_path( int argc, char** argv );
   static void dump_file_size(const char* path);
   static void dump_version();
   static std::string prepare_path(const char* dir_, const char* reldir_, const char* name ); 
};

namespace fs = boost::filesystem;

void UseBoostFS::dump_file_size(const char* path)
{
    std::cout << "UseBoostFS::dump_file_size: \"" << path  << "\" " << fs::file_size(path) << '\n';
}


const char* UseBoostFS::concat_path(int argc, char** argv)
{
    fs::path p ;
    for(int i=1 ; i < argc ; i++)
    {
         char* a = argv[i] ; 
         if(a) p /= a ; 
    }

    std::string x = p.string() ;
    return strdup(x.c_str());
}


void UseBoostFS::dump_version()
{
   std::cout 
          << "UseBoostFS::dump_version "         
          << BOOST_VERSION / 100000     << "."  // major version
          << BOOST_VERSION / 100 % 1000 << "."  // minor version
          << BOOST_VERSION % 100                // patch level
          << std::endl;
}


std::string UseBoostFS::prepare_path(const char* dir_, const char* reldir_, const char* name )
{   
    fs::path fdir(dir_);
    if(reldir_) fdir /= reldir_ ;

    if(!fs::exists(fdir))
    {   
        if (fs::create_directories(fdir))
        {   
            std::cout << "created directory " << fdir.string().c_str() << std::endl  ;
        }   
    }   

    fs::path fpath(fdir); 
    fpath /= name ;  

    return fpath.string();
}


int main(int argc, char** argv)
{
 
   const char* path =  argc < 2 ? argv[0] : UseBoostFS::concat_path( argc, argv ); 

   UseBoostFS::dump_file_size(path);
   UseBoostFS::dump_version();

   std::string p0 = UseBoostFS::prepare_path("/tmp/UseBoostFSManual", "red",  "name.txt" ); 
   std::cout << " p0 " << p0 << std::endl ;   

   std::string p1 = UseBoostFS::prepare_path("/tmp/UseBoostFSManual", "red", "name.txt" ); 
   std::cout << " p1 " << p1 << std::endl ;   

   return 0;

}

