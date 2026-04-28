openssl-env(){      echo -n ;  }
openssl-vi(){       vi $BASH_SOURCE ; }
openssl-usage(){ cat << EOU
openssl
========

openssl is required by libcurl

This script assumes sourcing by the openssl- precursor defined by
sourcing externals/externals.bash which means externals- bash
functions such as externals-base and externals-curl are available

Usage::

    source ~/opticks/externals/externals.bash
    openssl-
    openssl-info
    openssl--

During development update the bash functions with the precursor::

    openssl-


If suffering from slow network place the tarball into the cache directory::

    scp \$(openssl-srcball) O:
    # open cvmfs transaction and copy srcball into the download cache etc..

And define envvar pointing to the directory::

    export EXTERNALS_DOWNLOAD_CACHE=/cvmfs/opticks.ihep.ac.cn/opticks_download_cache


As openssl is a very widely used lib it would
not be useful to treat it as an Opticks managed
external with prefix OPTICKS_PREFIX/externals because
that would cause version inconsistency issues with other
versions of openssl that are in use by the tree
of all packages and externals being used.

Hence access to the openssl lib must be managed
as a "basis" external to the entire tree of packages
being built together, eg JUNOSW+Opticks.


EOU
}

openssl-info(){ cat << EOI

   openssl-name     : $(openssl-name)
   openssl-reldir   : $(openssl-reldir)
   externals-base   : $(externals-base)
   openssl-dir      : $(openssl-dir)
   openssl-idir     : $(openssl-idir)
   openssl-prefix   : $(openssl-prefix)
   openssl-url      : $(openssl-url)
   openssl-srcball  : $(openssl-srcball)
   openssl-binball  : $(openssl-binball)

EOI
}

openssl-reldir(){ echo openssl/$(openssl-name) ; }
openssl-dir(){    echo $(externals-base)/build/$(openssl-reldir) ; }
openssl-prefix(){ echo $(externals-base)/$(openssl-reldir) ; }
openssl-idir(){   echo $(openssl-prefix) ; }

openssl-cd(){  cd $(openssl-dir); }
openssl-icd(){ cd $(openssl-idir); }

openssl-version(){ echo 3.2.0 ; }
openssl-name(){    echo openssl-$(openssl-version) ; }  # NB no lib
openssl-url(){     echo https://github.com/openssl/openssl/releases/download/openssl-3.2.0/$(openssl-name).tar.gz ; }
openssl-srcball(){ echo $(dirname $(openssl-dir))/$(openssl-name).tar.gz ; }
openssl-binball(){ echo $(externals-base)/dist/$(openssl-name).tar.xz ; }



openssl-get()
{
   local dir=$(dirname $(openssl-dir)) &&  mkdir -p $dir && cd $dir

   local rc=0
   local url=$(openssl-url)
   local tgz=$(basename $url)
   local opt=$( [ -n "${VERBOSE}" ] && echo "-xzf" || echo "-xzvf" )

   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && externals-curl $url
   [ ! -d "$nam" ] && tar $opt $tgz
   [ ! -d "$nam" ] && rc=1

   return $rc
}

openssl-build()
{
    openssl-cd
    ./config \
       --prefix=$(openssl-prefix) \
       --openssldir=$(openssl-prefix)/ssl \
       shared \
       zlib

   #  -Wl,-rpath,'$(LIBRPATH)'


    # Build (OpenSSL 3.x is significantly faster with parallel jobs)
    make -j$(nproc)

    # Optional but recommended: run tests (takes a few minutes)
    # make test

    make install
}

openssl-dist()
{
    local prefix=$(openssl-prefix)
    local name=$(basename $prefix)
    local fold=$(dirname $prefix)
    local binball=$(openssl-binball)

    echo $BASH_SOURCE $FUNCNAME creating binball $binball from prefix $prefix name $name fold $fold
    if [ -f "$binball" ]; then
        echo BASH_SOURCE $FUNCNAME binball exists already
    else
        mkdir -p $(dirname $binball)
        tar -cvJf $binball -C $fold $name
        : J creates xz compressed tarball
    fi
}

openssl-dist-tvf()
{
    local binball=$(openssl-binball)
    tar tvf $binball
}


openssl--()
{
   local msg="=== $FUNCNAME :"
   openssl-get
   [ $? -ne 0 ] && echo $msg get FAIL && return 1

   openssl-build
   [ $? -ne 0 ] && echo $msg build FAIL && return 2

   openssl-bashrc
   [ $? -ne 0 ] && echo $msg bashrc FAIL && return 2

   openssl-dist
   [ $? -ne 0 ] && echo $msg dist FAIL && return 2

   return 0
}


openssl-bashrc()
{
    local path=$(openssl-prefix)/bashrc
    echo $BASH_SOURCE $FUNCNAME - writing $path - prefix needs to be named and versioned for this to make sense
    openssl-bashrc- > $path
}


openssl-bashrc-()
{
   cat << EOH
## generated $(date) by $(realpath $BASH_SOURCE) $FUNCNAME

HERE=\$(dirname \$(realpath \$BASH_SOURCE))

if [ -z "\${JUNOTOP}" ]; then
export JUNO_EXTLIB_openssl_HOME=\$HERE
else
export JUNO_EXTLIB_openssl_HOME=\$HERE
fi
EOH

cat << \EOS
export PATH=${JUNO_EXTLIB_openssl_HOME}/bin:${PATH}
if [ -d ${JUNO_EXTLIB_openssl_HOME}/lib ];
then
export LD_LIBRARY_PATH=${JUNO_EXTLIB_openssl_HOME}/lib:${LD_LIBRARY_PATH}
fi
if [ -d ${JUNO_EXTLIB_openssl_HOME}/lib/pkgconfig ];
then
export PKG_CONFIG_PATH=${JUNO_EXTLIB_openssl_HOME}/lib/pkgconfig:${PKG_CONFIG_PATH}
fi
if [ -d ${JUNO_EXTLIB_openssl_HOME}/lib64 ];
then
export LD_LIBRARY_PATH=${JUNO_EXTLIB_openssl_HOME}/lib64:${LD_LIBRARY_PATH}
fi
if [ -d ${JUNO_EXTLIB_openssl_HOME}/lib64/pkgconfig ];
then
export PKG_CONFIG_PATH=${JUNO_EXTLIB_openssl_HOME}/lib64/pkgconfig:${PKG_CONFIG_PATH}
fi
export CPATH=${JUNO_EXTLIB_openssl_HOME}/include:${CPATH}
export MANPATH=${JUNO_EXTLIB_openssl_HOME}/share/man:${MANPATH}

# For CMake search path
export CMAKE_PREFIX_PATH=${JUNO_EXTLIB_openssl_HOME}:${CMAKE_PREFIX_PATH}

EOS

}



openssl-wipe(){
  openssl-wipe-build
  openssl-wipe-prefix
}

openssl-wipe-build()
{
   local dir=$(openssl-dir)
   [ ! -d "$dir" ] && echo "$BASH_SOURCE $FUNCNAME dir $dir does not exist." && return

   local fold=$(dirname $dir)
   local name=$(basename $dir)

    pushd $fold
    if [ -d "$name" ]; then
        echo $BASH_SOURCE $FUNCNAME - removing name $name from fold $fold
        rm -rf "$name"
    else
        echo $BASH_SOURCE $FUNCNAME - NON-EXISTING name $name from fold $fold
    fi
    popd
}

openssl-wipe-prefix()
{
    local prefix=$(openssl-prefix)
    [ ! -d "$prefix" ] && echo "$BASH_SOURCE $FUNCNAME prefix $prefix does not exist." && return

    local fold=$(dirname $prefix)
    local name=$(basename $prefix)

    pushd $fold
    if [ -d "$name" ]; then
        echo $BASH_SOURCE $FUNCNAME - removing name $name from fold $fold
        rm -rf "$name"
    else
        echo $BASH_SOURCE $FUNCNAME - NON-EXISTING name $name from fold $fold
    fi
    popd
}

openssl-find()
{
    find $(openssl-prefix) -type f
}



