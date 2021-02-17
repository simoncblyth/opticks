oc-using-opticks-packages-without-cmake
=========================================


``opticks-config`` or ``oc`` are symbolic links to oc.bash
---------------------------------------------------------------

::

    epsilon:~ blyth$ which oc
    /usr/local/opticks/bin/oc

    epsilon:~ blyth$ l /usr/local/opticks/bin/ | grep oc
    lrwxr-xr-x  1 blyth  staff       7 Feb 17 14:03 opticks-config -> oc.bash
    lrwxr-xr-x  1 blyth  staff       7 Feb 17 14:03 oc -> oc.bash
    -rwxr-xr-x  1 blyth  staff   18206 May 16  2020 oc.bash


    epsilon:~ blyth$ oc -cflags SysRap
    -DOPTICKS_SYSRAP -DOPTICKS_OKCONF -I/usr/local/opticks/include/SysRap -I/usr/local/opticks/include/OKConf -I/usr/local/opticks/externals/plog/include -std=c++11



