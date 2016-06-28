#include <cstdlib>
#include <cstdio>


int main(int, char**, char** envp)
{
    while(*envp)
        printf("%s\n",*envp++);

    return 0 ; 
}
