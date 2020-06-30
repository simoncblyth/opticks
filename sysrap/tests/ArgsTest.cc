// gcc ArgsTest.cc -lstdc++ -Wvla -o /tmp/ArgsTest && /tmp/ArgsTest red green blue cyan magenta yellow black  
// head -1 ArgsTest.cc | perl -pe 's,//,,' - | sh  

#include <string.h>
#include <stdio.h>


void check(int argc, char** argv)
{
    char** args = new char*[argc] ; 
    for(int i=0 ; i < argc ; ++i ) args[i] = strdup(argv[i]) ; 
    for(int i=0 ; i < argc ; ++i ) printf(" %2d : [%s] \n", i, args[i] ); 
}

void check1()
{
    int argc(1);
    //  char* argv[argc] ;   // <-- ok with clang, but with juno options -Wvla :  warning: ISO C++ forbids variable length array ‘argv’

    char** argv = new char*[argc] ; 
    argv[0] = (char*)"ArgsTest.cc" ; 
    check(argc, argv);  
    delete [] argv ; 

}

void check1fix()
{
    //enum { argc = 1 } ;
    const int argc = 1 ; 
    char* argv[argc] ; 
    argv[0] = (char*)"ArgsTest.cc" ;  
    check(argc, argv);  
}


void check2()
{

    //int argc(2); 
    //char* argv[argc] ; 
    enum { argc = 2 } ;
    char* argv[argc] ; 

    argv[0] = (char*)"ArgsTest.cc" ; 
    argv[1] = (char*)"red" ; 

    check(argc, argv);  
}

int main(int argc, char** argv)
{
    //check(argc, argv); 

    //check1(); 
    check1fix(); 
    check2(); 

    return 0 ; 
}
