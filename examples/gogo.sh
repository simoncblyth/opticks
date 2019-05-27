#!/bin/bash 

gogo-find(){ find . -name go.sh ; }
gogo-all(){
    local go
    gogo-find | while read go ; do 
        gogo-one $go
    done
}


gogo-one-()
{
    local rc
    ./go.sh > /dev/null 2>&1
    rc=$?
    printf "%-40s %d \n" $go $rc 
}

gogo-one(){
    local go=$1 
    local iwd=$PWD

    local dir=$(dirname $go)
    cd $dir 

    printf "%-40s \n" $go 

    if [ -f "SKIP" ]; then
        printf "%-40s %s %s \n" $go "SKIP" "$(head -1 SKIP)"
    else 
        gogo-one-  
    fi 

    cd $iwd
}

gogo-all


