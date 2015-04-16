#include <iostream>
#include <boost/filesystem.hpp>
using namespace std;
//using namespace boost::filesystem;

namespace fs = boost::filesystem;



int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    cout << "Usage: tut2 path\n";
    return 1;
  }

  fs::path p (argv[1]);   // p reads clearer than argv[1] in the following code

  if (fs::exists(p))    // does p actually exist?
  {
    if (fs::is_regular_file(p))        // is p a regular file?
      cout << p << " size is " << fs::file_size(p) << '\n';

    else if (fs::is_directory(p))      // is p a directory?
      cout << p << " is a directory\n";

    else
      cout << p << " exists, but is neither a regular file nor a directory\n";
  }
  else
    cout << p << " does not exist\n";

  return 0;
}
