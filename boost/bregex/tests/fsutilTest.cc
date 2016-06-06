#include "fsutil.hh"
#include "dbg.hh"

int main(int argc, char** argv)
{

   DBG(argv[0]," fsutil::FormPath(\"/tmp\") ", fsutil::FormPath("/tmp") ); 
   DBG(argv[0]," fsutil::FormPath(\"/tmp\",\"name.npy\") ", fsutil::FormPath("/tmp","name.npy") ); 
   DBG(argv[0]," fsutil::FormPath(\"/tmp\",\"sub\",\"name.npy\") ", fsutil::FormPath("/tmp","sub","name.npy") ); 


   fsutil::CreateDir("/tmp/a/b/c");


   return 0 ; 
}



