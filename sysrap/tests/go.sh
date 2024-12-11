#!/bin/bash
usage(){ cat << EOU

~/o/sysrap/tests/go.sh 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

tests=$(sed 's/#.*//' < tests.txt)

count=0
pass=0
fail=0

flog=""
t0="$(printf "\n$(date)"$'\n')"
echo "$t0"
echo 

for t in $tests ; do 

    tt=$(realpath $t)

    l0="$(printf " === %0.3d === [ $tt "$'\n' "$count")"
    echo "$l0"
    eval $tt > /dev/null 2>&1
    rc=$?

    if [ $rc -ne 0 ]; then 
        msg="***FAIL***"
        fail=$(( $fail + 1 ))
    else
        msg=PASS
        pass=$(( $pass + 1 ))
    fi   
    l1="$(printf " === %0.3d === ] %s "$'\n' "$count" "$msg")"
    echo "$l1"
    echo
 
    if [ $rc -ne 0 ]; then
       flog+="$l0"$'\n'
       flog+="$l1"$'\n'
    fi 

   #[ $rc -ne 0 ] && echo non-zero RC && break
    count=$(( $count + 1 ))
done 

t1="$(printf "\n$(date)"$'\n')"

echo "$t0"
echo "$t1"
echo

printf " TOTAL : %d \n" $count
printf " PASS  : %d \n" $pass
printf " FAIL  : %d \n" $fail

echo "$flog"


