#/bin/bash -l 

name=G4OrbTest
MODE=config

if [ "$(uname)" == "Darwin" -a "$MODE" == "Manual" ]; then 

    clhep_prefix=/usr/local/opticks_externals/clhep_2440
    clhep_libdir=$clhep_prefix/lib
    clhep_incdir=$clhep_prefix/include

    g4_prefix=/usr/local/opticks_externals/g4_1042
    g4_libdir=$g4_prefix/lib
    g4_incdir=$g4_prefix/include/Geant4

    gcc $name.cc \
          -std=c++11 \
          -I$clhep_incdir \
          -L$clhep_libdir \
          -lCLHEP \
          -I$g4_incdir \
          -L$g4_libdir \
           -lG4global \
           -lG4geometry \
           -lstdc++  \
           -o /tmp/$name 

    [ $? -ne 0 ] && echo compile error && exit 1

elif [ "$(uname)" == "Linux" -a "$MODE" == "Manual" ]; then

    clhep_prefix=/data/blyth/junotop/ExternalLibs/CLHEP/2.4.1.0
    clhep_libdir=$clhep_prefix/lib
    clhep_incdir=$clhep_prefix/include

    g4_prefix=/data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno
    g4_libdir=$g4_prefix/lib64
    g4_incdir=$g4_prefix/include/Geant4

    gcc $name.cc \
          -DG4USE_STD11 \
          -std=c++11 \
          -I$clhep_incdir \
          -L$clhep_libdir \
          -lCLHEP \
          -I$g4_incdir \
          -L$g4_libdir \
           -lG4global \
           -lG4geometry \
           -lstdc++  \
            -lpthread \
           -o /tmp/$name 

    [ $? -ne 0 ] && echo compile error && exit 1


    /tmp/$name
    [ $? -ne 0 ] && echo run error && exit 2

else

    echo geant4-config mode using 
    geant4-config --cflags 
    geant4-config --libs

    gcc $name.cc \
          $(geant4-config --cflags) \
          $(geant4-config --libs) \
          -lstdc++ \
          -o /tmp/$name
    [ $? -ne 0 ] && echo compile error && exit 1

    /tmp/$name
    [ $? -ne 0 ] && echo run error && exit 2

fi



exit 0 




