#!/bin/bash -l 
msg="=== $BASH_SOURCE: "
name=UsePthreadStandalone
tmpdir=/tmp/$USER/opticks/$name
mkdir -p $tmpdir && cd $tmpdir 

arg=${1:-0}

if [ "$arg" == "0" ]; then 

cat << EOS0 > $name.cc
// https://www.geeksforgeeks.org/multithreading-c-2/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> 
#include <pthread.h>

void *myThreadFun(void *vargp)
{
sleep(1);
printf("printf from Thread \n");
return NULL;
}

int main()
{
pthread_t thread_id;
printf("printf before Thread\n");
pthread_create(&thread_id, NULL, myThreadFun, NULL);
pthread_join(thread_id, NULL);
printf("printf after thread\n");
exit(0);
}

EOS0


elif [ "$arg" == "1" ]; then 

cat << EOS1 > $name.cc

#include <iostream>
#include <unistd.h> 
#include <pthread.h>

void *myThreadFun(void *vargp)
{
sleep(1);
std::cout << " std::cout from Thread " << std::endl ;
return NULL;
}

int main()
{
pthread_t thread_id;
std::cout << " std::cout before thread " << std::endl ; 
pthread_create(&thread_id, NULL, myThreadFun, NULL);
pthread_join(thread_id, NULL);
std::cout << " std::cout after thread " << std::endl ; 
exit(0);
}

EOS1

fi 


cat $name.cc

gcc $name.cc -std=c++11 -lstdc++ -lpthread -o $name
[ $? -ne 0 ] && echo $msg compilation error && exit 1

./$name
[ $? -ne 0 ] && echo $msg run error && exit 2

exit 0 

