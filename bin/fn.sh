#!/usr/bin/env bash 

notes(){ cat << EON


* https://stackoverflow.com/questions/4303128/how-to-use-multiple-arguments-for-awk-with-a-shebang-i-e

Normal approach of using "bash -l" to source the 
login scripts doesnt work on Linux together with shebang
(it does work on macOS):: 

     #!/usr/bin/env bash -l

So in the below find OPTICKS_HOME from the path to this script.
But this is not a practical general solution as users exporting 
OPTICKS_KEY in .bashrc to pick a geometry is expected usage.

EON
}


export OPTICKS_HOME=$(dirname $(dirname $0))
opticks-(){ . $OPTICKS_HOME/opticks.bash && opticks-env $* ; } 
opticks-

fn-

cmd="fn-lv $*"

echo $cmd
eval $cmd
rc=$?

echo $0 rc $rc
exit $rc

