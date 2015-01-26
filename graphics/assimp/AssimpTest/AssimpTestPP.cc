#include "AScene.hh"

int main(int argc, char** argv)
{
   const char* key = "DAE_NAME_DYB_NOEXTRA";
   const char* path = getenv(key);
   if(!path) return 1 ; 

   AScene a(path);
   a.Dump();

   const char* query = (argc > 1 )? argv[1] : "/"  ;

   printf("query %s \n", query ); 

   aiNode* node = a.searchNode(query);

   printf("node %p \n", node); 



   return 0 ;
}
