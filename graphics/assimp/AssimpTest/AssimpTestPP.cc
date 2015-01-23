#include "AScene.hh"

int main(int argc, char** argv)
{
   const char* key = "DAE_NAME_DYB_NOEXTRA";
   const char* path = getenv(key);
   if(!path) return 1 ; 

   AScene a(path);
   a.Dump();


   return 0 ;
}
