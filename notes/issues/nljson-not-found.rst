nljson-not-found
==================

New required external.::

    o
    om
    ...


    -- Up-to-date: /home/blyth/local/opticks/lib/SPackTest
    -- Up-to-date: /home/blyth/local/opticks/lib/SBitTest
    -- Up-to-date: /home/blyth/local/opticks/lib/SRandTest
    -- Up-to-date: /home/blyth/local/opticks/lib/SRngSpecTest
    === om-make-one : boostrap        /home/blyth/opticks/boostrap                                 /home/blyth/local/opticks/build/boostrap                     
    -- Configuring BoostRap
    -- BoostAsio_FOUND : YES
    -- OPTICKS_PREFIX           : /home/blyth/local/opticks
    -- NLJSON_MODULE            : /home/blyth/opticks/cmake/Modules/FindNLJSON.cmake
    -- NLJSON_INCLUDE_DIR       : NLJSON_INCLUDE_DIR-NOTFOUND 
    -- NLJSON_FOUND             : NO
    CMake Error at /home/blyth/opticks/cmake/Modules/FindNLJSON.cmake:25 (message):
      NLJSON NOT FOUND
    Call Stack (most recent call first):
      CMakeLists.txt:17 (find_package)


    -- Configuring incomplete, errors occurred!
    See also "/home/blyth/local/opticks/build/boostrap/CMakeFiles/CMakeOutput.log".
    See also "/home/blyth/local/opticks/build/boostrap/CMakeFiles/CMakeError.log".
    make: *** [cmake_check_build_system] Error 1
    === om-one-or-all make : non-zero rc 2
    === om-all om-make : ERROR bdir /home/blyth/local/opticks/build/boostrap : non-zero rc 2
    [blyth@localhost opticks]$ 



Install the new required external::    

    [blyth@localhost opticks]$ nljson-
    [blyth@localhost opticks]$ nljson--
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   632  100   632    0     0    841      0 --:--:-- --:--:-- --:--:--   842
    100  904k  100  904k    0     0   251k      0  0:00:03  0:00:03 --:--:--  372k
    [blyth@localhost nljson]$ 


Proceed with build::

    o
    om






