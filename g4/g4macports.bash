# === func-gen- : g4/g4macports fgp g4/g4macports.bash fgn g4macports fgh g4
g4macports-src(){      echo g4/g4macports.bash ; }
g4macports-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4macports-src)} ; }
g4macports-vi(){       vi $(g4macports-source) ; }
g4macports-env(){      elocal- ; }
g4macports-usage(){ cat << EOU


Macports Geant4
-----------------

::

    simon:env blyth$ port info geant4.10.1
    geant4.10.1 @4.10.01.p02 (science)
    Variants:             clhep, debug, examples, [+]gdml, motif_x11, opengl_x11, [+]qt, qt5, raytracer_x11, threads, universal

    Description:          Geant4 is a toolkit for the simulation of the passage of particles through matter. Its areas of application include high energy, nuclear and
                          accelerator physics, as well as studies in medical and space science. The two main reference papers for Geant4 are published in Nuclear
                          Instruments and Methods in Physics Research A 506 (2003) 250-303, and IEEE Transactions on Nuclear Science 53 No. 1 (2006) 270-278.
    Homepage:             http://geant4.web.cern.ch/

    Build Dependencies:   cmake, pkgconfig
    Library Dependencies: geant4.10.1-data, expat, zlib, qt4-mac, xercesc3
    Runtime Dependencies: geant4_select
    Platforms:            darwin
    License:              Restrictive/Distributable
    Maintainers:          mojca@macports.org, openmaintainer@macports.org
    simon:env blyth$ 


    https://guide.macports.org/chunked/using.variants.html
   
    Variants marked with "+" are included by default, can negate these with a "-"
 

    simon:workflow blyth$ port variants geant4.10.1
    geant4.10.1 has the variants:
       clhep: Use external clhep
       debug
       examples: Build and install examples (not recommended)
    [+]gdml: Build with Geometry Description Markup Language (GDML)
       motif_x11: Build with Motif (X11) user interface and visualization driver
       opengl_x11: Build with X11 visualisation drivers
    [+]qt: Build with Qt 4 support
         * conflicts with qt5
       qt5: Build with Qt 5 support
         * conflicts with qt
       raytracer_x11: Build with Raytracer (X11) visualization driver
       threads: Build with multi-threading support
       universal: Build for multiple architectures
    simon:workflow blyth$ 








EOU
}
g4macports-dir(){ echo $(local-base)/env/g4/g4-g4macports ; }
g4macports-cd(){  cd $(g4macports-dir); }
g4macports-mate(){ mate $(g4macports-dir) ; }
g4macports-get(){
   local dir=$(dirname $(g4macports-dir)) &&  mkdir -p $dir && cd $dir

}
