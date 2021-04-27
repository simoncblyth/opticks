// name=getenvTest ; gcc $name.cc -lstdc++ -std=c++11 -o /tmp/$name && /tmp/$name
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv)
{
    const char* key = argc > 1 ? argv[1] : "HOME" ; 
    char* val = getenv(key) ; 
    printf("%s\n",val?val:"-"); 
    return 0 ; 
}
