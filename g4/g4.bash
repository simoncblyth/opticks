# === func-gen- : g4/g4 fgp g4/g4.bash fgn g4 fgh g4
g4-src(){      echo g4/g4.bash ; }
g4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4-src)} ; }
g4-vi(){       vi $(g4-source) ; }
g4-env(){      elocal- ; }
g4-usage(){ cat << EOU


Geant4 10.2, December 4th, 2015
----------------------------------

* https://geant4.web.cern.ch/geant4/support/ReleaseNotes4.10.2.html
* https://geant4.web.cern.ch/geant4/support/download.shtml
* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/


CMake 3.3 or higher
~~~~~~~~~~~~~~~~~~~~

Minimum required version of CMake to build Geant4 is 3.3. User applications can
still use CMake 2.8.X or above for configuration and compilation. It is however
recommended to migrate to CMake 3.3 or above for its improved support of C++
compile and target features.

Installed via macports, annoyingly required to install java too and entailed
rebuilding several other packages including python, mysql, llvm see ~/macports/cmake33.log


Xcode 6 or higher
~~~~~~~~~~~~~~~~~~




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

g4-name(){ echo geant4.10.02 ; } 
g4-dir(){  echo $(local-base)/env/g4/$(g4-name) ; }
g4-edir(){ echo $(env-home)/g4 ; }
g4-cd(){   cd $(g4-dir); }
g4-ecd(){  cd $(g4-edir); }

g4-url(){ echo http://geant4.cern.ch/support/source/$(g4-name).tar.gz ; }
g4-get(){
   local dir=$(dirname $(g4-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(g4-url)
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -L -O $url 
   [ ! -d "$nam" ] && tar zxvf $tgz 
}




