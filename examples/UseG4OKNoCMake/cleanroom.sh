#!/usr/bin/env -i bash 

notes(){ cat << EON

env -i on the shebang line works, BUT 
not very useful as need to find the installation.
via some very minimal environment at the least.

Unless can discern that from the path to the script, 
meaning the script needs to be installed as part 
of the distro.

EON
}

echo cleanroom env \$0 : $0
env 

