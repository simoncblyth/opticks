# === func-gen- : boost/boostlog fgp boost/boostlog.bash fgn boostlog fgh boost
boostlog-src(){      echo boost/boostlog.bash ; }
boostlog-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(boostlog-src)} ; }
boostlog-vi(){       vi $(boostlog-source) ; }
boostlog-env(){      elocal- ; }
boostlog-usage(){ cat << EOU

boost log
~~~~~~~~~~~

http://boost-log.sourceforge.net/libs/log/doc/html/index.html  
http://stackoverflow.com/questions/6076405/what-is-boost-log-how-to-get-it-and-how-to-build-it
https://svn.boost.org/trac/boost/wiki/ReviewScheduleLibraries#Boost.Log
http://lists.boost.org/boost-announce/2010/03/0256.php
    laundry list of changes from the boost review



   http://boost-log.sourceforge.net/libs/log/doc/html/index.html
       early version of docs


EOU
}
boostlog-dir(){ echo /tmp/env/boost/$(boostlog-name) ; }
boostlog-cd(){  cd $(boostlog-dir); }
boostlog-mate(){ mate $(boostlog-dir) ; }
boostlog-name(){ echo boost-log2-snapshot-628 ; }
boostlog-url(){  echo http://downloads.sourceforge.net/project/boost-log/$(boostlog-name).zip ; }
boostlog-get(){
   local dir=$(dirname $(boostlog-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(boostlog-url)
   local zip=$(basename $url)
   local nam=${zip/.zip}
   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -d "$nam" ] && unzip $zip
}
