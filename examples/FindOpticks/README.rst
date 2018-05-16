
Example of building an executable that uses Opticks using the CMake find_package mechanism via OpticksConfig.cmake::

    cd $HOME/opticks/examples/FindOpticks

    sdir=$(pwd) && name=$(basename $sdir) &&  bdir=/tmp/$USER/$name/build && mkdir -p $bdir && cd $bdir && pwd && cmake -DOpticks_DIR=/usr/local/opticks/config $sdir 

    make

    make install   # installs executable to  /usr/local/lib/FindOpticks




