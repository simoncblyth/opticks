# === func-gen- : xml/xercesc/xercesc fgp externals/xercesc.bash fgn xercesc fgh xml/xercesc
xercesc-src(){      echo externals/xercesc.bash ; }
xercesc-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(xercesc-src)} ; }
xercesc-vi(){       vi $(xercesc-source) ; }
xercesc-env(){      elocal- ; }
xercesc-usage(){ cat << EOU

XERCESC
========

Issues
-------

* https://issues.apache.org/jira/browse/XERCESC/?selectedTab=com.atlassian.jira.jira-projects-plugin:summary-panel

Windows VS 2015 ?
------------------

* http://comments.gmane.org/gmane.text.xml.xerces-c.devel/11718

Patch for VS 2015 ?
~~~~~~~~~~~~~~~~~~~~~

* see ome- for application of the patch

* http://article.gmane.org/gmane.text.xml.xerces-c.devel/11822/match=cmake

::

    On 2016-05-02 09:11, Heiko Nardmann wrote:
    > Hi together!
    > 
    > Did someone manage to get xerces build with MSVC 2015?

    Yes.  See

    https://github.com/ome/ome-cmake-superbuild/tree/develop/packages/xerces/patches

    You want win-vc14.diff.  Note this also has the ICU fix which is in the 
    win-vc12.diff patch; this applies to all MSVC versions.  I thought I'd 
    attached this patch to a ticket already, but I can't find it.  Either I 
    didn't or it's not findable!  Either way, if anyone with access would 
    care to apply this to each VC version that would be appreciated.  And if 
    you want to pick up the VC14 support, that would likewise be 
    appreciated.

    Regards,
    Roger


* https://github.com/ome/ome-cmake-superbuild


Kind offer falls on deaf ears
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Roger Leigh | 23 Jun 17:08 2015 | CMake support::

    Firstly, hello.  I am Roger Leigh, a C++ developer currently working on 
    a project (Bio-Formats-C++) which makes use of Xerces-C++.

    One of the parts of this role is integrating several upstream projects, 
    of which xerces is one, into a larger project which needs to build on 
    Unix/Linux/Windows.  While the xerces-c project provides an 
    autotools-based build and several different visual studio solution 
    files, I was wondering if you had considered the use of a tool such as 
    CMake, which can generate solution files for all visual studio versions 
    (including 2015), Makefiles, and project files for a number of IDEs, 
    including Eclipse?  This allows all the platforms to be supported well 
    from a common set of build rules, and means you don't need to maintain 
    separate solutions for each visual studio release.

    The reason for asking is that over the course of the last few weeks, 
    I've converted several open source projects from autotools+separate msvc 
    builds to a unified cmake build and submitted these to their upstream 
    developers.  If this is something you would find beneficial and useful, 
    then I would be happy to do the same for xerces-c.  This can, of course, 
    co-exist with the existing build systems.

    Kind regards,
    Roger Leigh

    ...

    In the interim, I've been trying to use the provided VC solution/project
    files, and I've run into some problems.  The "ICU Debug" and "ICU Release"
    configurations are broken for all the VC versions I've looked at (10, 11
    and 12), likely applicable to all versions.  They don't link against the
    libicuuc[d].lib libraries for any of the x64 platform variants.  And they
    don't link against the debug library for the debug configuration variants.

    The following patch demonstrates a possible fix for VC12, which should
    apply to all previous versions as well.

    Regarding CMake support, this discrepancy could have been easily avoided
    by having a simple feature test and option for ICU support, rather than a
    combinatorial explosion of configurations, platforms and VC versions.  My
    offer to add such support still stands, should you wish to take advantage
    of it.

    Kind regards,
    Roger





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

xercesc-prefix(){  
  case $(uname -s) in 
      Darwin) echo /opt/local ;;
    MINGW64*) echo /mingw64 ;;
           *) echo /usr/local ;;
  esac  
}


xercesc-include-dir(){ 
  if [ "$NODE_TAG" == "MGB" ]; then
      ome-
      ome-xercesc-include
  else
      echo $(xercesc-prefix)/include ;
  fi
}
xercesc-library(){  

  if [ "$NODE_TAG" == "MGB" ]; then
      ome-
      ome-xercesc-lib
  else 
      case $(uname -s) in 
        Darwin) echo    $(xercesc-prefix)/lib/libxerces-c.dylib    ;;
         MINGW64*) echo $(xercesc-prefix)/bin/libxerces-c-3-1.dll  ;;
      esac
  fi 
}

xercesc-geant4-export(){
  export XERCESC_INCLUDE_DIR=$(xercesc-include-dir)
  export XERCESC_LIBRARY=$(xercesc-library)
}



