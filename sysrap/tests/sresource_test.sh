#!/bin/bash
usage(){ cat << EOU

~/o/sysrap/tests/sresource_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=sresource_test
tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name

defarg="info_gcc_run"
arg=${1:-$defarg}

vv="BASH_SOURCE name defarg arg PWD tmp TMP FOLD bin"

if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/gcc}" != "$arg" ]; then
    gcc $name.cc -std=c++17 -lstdc++ -I.. -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE gcc error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [ "${arg/lim}" != "$arg" ]; then

cmds() {
    cat <<'EOC'
whoami
ulimit -l
ulimit -H -l
EOC
}
    while IFS= read -r cmd; do
        [[ -z "$cmd" ]] && continue
        printf '%s\n' "$cmd"
        eval "$cmd"
        echo
    done < <(cmds)

fi


exit 0

