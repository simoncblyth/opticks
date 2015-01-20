/*
  g++ -I/opt/local/include -c dirlist.cc 
  g++ -L/opt/local/lib -lboost_filesystem -lboost_system dirlist.o -o dirlist 


  http://www.boost.org/doc/libs/1_49_0/libs/filesystem/v3/doc/index.htm
  http://www.boost.org/doc/libs/1_49_0/libs/filesystem/v3/doc/tutorial.html


  gets exception and aborts on permission denied 

 */
#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>

using namespace boost::filesystem; 
using namespace std ;

void show_files( const path& directory, bool recurse_into_subdirs = true )
{
   try
   {	   
      if( exists( directory ) )
      {
          directory_iterator end ;
          for( directory_iterator iter(directory) ; iter != end ; ++iter ){
	      if(is_directory(*iter)){

	          cout << *iter << " (directory)\n" ;
  		  if( recurse_into_subdirs ) show_files(*iter) ;

	       } else {	   

	  	  cout << *iter << " (file)\n" ;
	       }
           }	   
       }
   }
    catch (const filesystem_error& ex)
    {
	cout << ex.what() << '\n';
    }
 

}

int main()
{
show_files( "/tmp" ) ;
}


