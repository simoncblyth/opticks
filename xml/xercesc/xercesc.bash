# === func-gen- : xml/xercesc/xercesc fgp xml/xercesc/xercesc.bash fgn xercesc fgh xml/xercesc
xercesc-src(){      echo xml/xercesc/xercesc.bash ; }
xercesc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(xercesc-src)} ; }
xercesc-vi(){       vi $(xercesc-source) ; }
xercesc-env(){      elocal- ; }
xercesc-usage(){ cat << EOU

XERCESC
========

D : macports install Oct 2014
---------------------------------

::

    (chroma_env)delta:geant4.9.5.p01 blyth$ port info xercesc
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    xercesc @2.8.0_3 (textproc)
    Variants:             universal

    Description:          Xerces-C++ is a validating XML parser written in a portable subset of C++. Xerces-C++ makes it easy to give your application the ability to read and write XML
                          data. A shared library is provided for parsing, generating, manipulating, and validating XML documents.
    Homepage:             http://xerces.apache.org/xerces-c/

    Conflicts with:       xercesc3
    Platforms:            darwin
    License:              Apache-2
    Maintainers:          chris.ridd@isode.com
    (chroma_env)delta:geant4.9.5.p01 blyth$ 


    delta:~ blyth$ sudo port install -v xercesc
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    --->  Fetching archive for xercesc
    --->  Attempting to fetch xercesc-2.8.0_3.darwin_13.x86_64.tbz2 from http://jog.id.packages.macports.org/macports/packages/xercesc
    --->  Attempting to fetch xercesc-2.8.0_3.darwin_13.x86_64.tbz2.rmd160 from http://jog.id.packages.macports.org/macports/packages/xercesc
    --->  Installing xercesc @2.8.0_3
    --->  Activating xercesc @2.8.0_3
    --->  Cleaning xercesc
    --->  Updating database of binaries
    --->  Scanning binaries for linking errors
    --->  No broken files found.                             
    delta:~ blyth$ 

    delta:~ blyth$ port contents xercesc | grep DOM.hpp
      /opt/local/include/xercesc/dom/DOM.hpp
      /opt/local/include/xercesc/dom/deprecated/DOM.hpp

    delta:~ blyth$ port contents xercesc | grep libxerces-c.dylib
      /opt/local/lib/libxerces-c.dylib




geant4 cmake options
--------------------

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/ch02s03.html



EOU
}
xercesc-dir(){ echo $(local-base)/env/xml/xercesc/xml/xercesc-xercesc ; }
xercesc-cd(){  cd $(xercesc-dir); }
xercesc-mate(){ mate $(xercesc-dir) ; }
xercesc-get(){
   local dir=$(dirname $(xercesc-dir)) &&  mkdir -p $dir && cd $dir

}

xercesc-include-dir(){ echo /opt/local/include ; }
xercesc-library(){  echo /opt/local/lib/libxerces-c.dylib ; }

xercesc-geant4-export(){
  [ "$NODE_TAG" != "D" ] && return  
  export XERCESC_INCLUDE_DIR=$(xercesc-include-dir)
  export XERCESC_LIBRARY=$(xercesc-library)
}
