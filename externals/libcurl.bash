libcurl-env(){      echo -n ;  }
libcurl-vi(){       vi $BASH_SOURCE ; }
libcurl-usage(){ cat << EOU
libcurl
=======

libcurl requires openssl

This script assumes sourcing by the libcurl- precursor defined by
sourcing externals/externals.bash which means externals- bash
functions such as externals-base and externals-curl are available

Usage, from a clean environment::

    source ~/opticks/externals/externals.bash
    openssl-
    openssl-info
    openssl--

    libcurl-
    libcurl-info
    libcurl--

During development update the bash functions with the precursor::

    libcurl-

If suffering from slow network place the tarball into the cache directory::

    scp \$(libcurl-srcball) O:
    # open cvmfs transaction and copy srcball into the download cache etc..

And define envvar pointing to the directory::

    export EXTERNALS_DOWNLOAD_CACHE=/cvmfs/opticks.ihep.ac.cn/opticks_download_cache



As libcurl is widely used it would not be useful to treat
it as an Opticks managed external with prefix OPTICKS_PREFIX/externals
because that would cause version inconsistency issues with other
versions of libcurl that are in use by the tree of all packages
and externals being used.

Hence access to the libcurl lib must be managed
as a "basis" external to the entire tree of packages
being built together, eg JUNOSW+Opticks.

Checking usage::

    ## careful to use clean env without "lco" conda, as that brings in another libcurl
    source /usr/local/ExternalLibs/openssl/openssl-3.2.0/bashrc
    source /usr/local/ExternalLibs/libcurl/curl-8.12.1/bashrc
    env | grep JUNO
    ldd $JUNO_EXTLIB_libcurl_HOME/lib64/libcurl.so

EOU
}

libcurl-deps(){
   : check that all deps other than libz and libc are controlled
   source $(externals-base)/openssl/openssl-3.2.0/bashrc
   source $(externals-base)/libcurl/curl-8.12.1/bashrc
   env | grep JUNO
   ldd $JUNO_EXTLIB_libcurl_HOME/lib64/libcurl.so
}


libcurl-info(){ cat << EOI

   libcurl-name    : $(libcurl-name)    # NB no lib
   libcurl-reldir  : $(libcurl-reldir)
   externals-base  : $(externals-base)
   libcurl-dir     : $(libcurl-dir)
   libcurl-bdir    : $(libcurl-bdir)
   libcurl-idir    : $(libcurl-idir)
   libcurl-prefix  : $(libcurl-prefix)
   libcurl-url     : $(libcurl-url)
   libcurl-srcball : $(libcurl-srcball)
   libcurl-binball : $(libcurl-binball)

EOI
}

libcurl-version(){ echo 8.12.1 ; }
libcurl-name(){    echo curl-$(libcurl-version) ; }  # NB no lib
libcurl-url(){     echo https://curl.se/download/$(libcurl-name).tar.gz ; }
libcurl-reldir(){ echo libcurl/$(libcurl-name)  ; }
libcurl-dir(){  echo $(externals-base)/build/$(libcurl-reldir) ; }
libcurl-bdir(){ echo $(libcurl-dir).build ; }
libcurl-prefix(){ echo $(externals-base)/$(libcurl-reldir) ; }
libcurl-idir(){   echo $(libcurl-prefix) ; }

libcurl-cd(){  cd $(libcurl-dir); }
libcurl-bcd(){ cd $(libcurl-bdir); }
libcurl-icd(){ cd $(libcurl-idir); }

libcurl-srcball(){ echo $(dirname $(libcurl-dir))/$(libcurl-name).tar.gz ; }
libcurl-binball(){ echo $(externals-base)/dist/$(libcurl-name).tar.xz ; }


libcurl-get(){
   local dir=$(dirname $(libcurl-dir)) &&  mkdir -p $dir && cd $dir

   local rc=0
   local url=$(libcurl-url)
   local tgz=$(basename $url)
   local opt=$( [ -n "${VERBOSE}" ] && echo "-xzf" || echo "-xzvf" )

   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && externals-curl $url
   [ ! -d "$nam" ] && tar $opt $tgz
   [ ! -d "$nam" ] && rc=1

   return $rc
}



libcurl-build()
{
   : unused features are disabled to reduce dependencies

   local iwd=$PWD
   local sdir=$(libcurl-dir)
   local bdir=$(libcurl-bdir)
   mkdir -p $bdir

   openssl-

   cd $bdir
   cmake $sdir \
    -DCMAKE_INSTALL_PREFIX=$(libcurl-prefix) \
    -DOPENSSL_ROOT_DIR=$(openssl-prefix) \
    -DCMAKE_BUILD_TYPE=Release \
    -DCURL_USE_LIBPSL=OFF \
    -DCURL_ZSTD=OFF \
    -DCURL_BROTLI=OFF \
    -DUSE_LIBIDN2=OFF \
    -DCURL_DISABLE_LDAP=ON \
    -DCURL_DISABLE_LDAPS=ON \
    -DCURL_USE_OPENSSL=ON \
    -DBUILD_CURL_EXE=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_SKIP_INSTALL_RPATH=ON \
    -DCMAKE_SKIP_RPATH=ON

   [ $? -ne 0 ] && echo $BASH_SOURCE $FUNCNAME - ERROR FROM cmake && return 1


   make -j$(nproc)
   [ $? -ne 0 ] && echo $BASH_SOURCE $FUNCNAME - ERROR FROM make && return 1

   make install
   [ $? -ne 0 ] && echo $BASH_SOURCE $FUNCNAME - ERROR FROM make install && return 1

   return 0
}


libcurl-dist()
{
    local prefix=$(libcurl-prefix)
    local name=$(basename $prefix)
    local fold=$(dirname $prefix)
    local binball=$(libcurl-binball)

    echo $BASH_SOURCE $FUNCNAME creating binball $binball from prefix $prefix name $name fold $fold
    if [ -f "$binball" ]; then
        echo BASH_SOURCE $FUNCNAME binball exists already
    else
        mkdir -p $(dirname $binball)
        tar -cvJf $binball -C $fold $name
        : J creates xz compressed tarball
    fi
}

libcurl-dist-tvf()
{
    local binball=$(libcurl-binball)
    tar tvf $binball
}



libcurl--()
{
   local msg="=== $FUNCNAME :"
   libcurl-get
   [ $? -ne 0 ] && echo $msg get FAIL && return 1

   libcurl-build
   [ $? -ne 0 ] && echo $msg build FAIL && return 2

   libcurl-bashrc
   [ $? -ne 0 ] && echo $msg bashrc FAIL && return 2

   libcurl-dist
   [ $? -ne 0 ] && echo $msg dist FAIL && return 2

   return 0
}


libcurl-bashrc()
{
   local path=$(libcurl-prefix)/bashrc
   echo $BASH_SOURCE $FUNCNAME - writing $path - prefix needs to be named and versioned for this to make sense
   libcurl-bashrc- > $path
}


libcurl-bashrc-()
{
   cat << EOH
## generated $(date) by $(realpath $BASH_SOURCE) $FUNCNAME

HERE=\$(dirname \$(realpath \$BASH_SOURCE))

if [ -z "\${JUNOTOP}" ]; then
export JUNO_EXTLIB_libcurl_HOME=\$HERE
else
export JUNO_EXTLIB_libcurl_HOME=\$HERE
fi
EOH

cat << \EOS
export PATH=${JUNO_EXTLIB_libcurl_HOME}/bin:${PATH}
if [ -d ${JUNO_EXTLIB_libcurl_HOME}/lib ];
then
export LD_LIBRARY_PATH=${JUNO_EXTLIB_libcurl_HOME}/lib:${LD_LIBRARY_PATH}
fi
if [ -d ${JUNO_EXTLIB_libcurl_HOME}/lib/pkgconfig ];
then
export PKG_CONFIG_PATH=${JUNO_EXTLIB_libcurl_HOME}/lib/pkgconfig:${PKG_CONFIG_PATH}
fi
if [ -d ${JUNO_EXTLIB_libcurl_HOME}/lib64 ];
then
export LD_LIBRARY_PATH=${JUNO_EXTLIB_libcurl_HOME}/lib64:${LD_LIBRARY_PATH}
fi
if [ -d ${JUNO_EXTLIB_libcurl_HOME}/lib64/pkgconfig ];
then
export PKG_CONFIG_PATH=${JUNO_EXTLIB_libcurl_HOME}/lib64/pkgconfig:${PKG_CONFIG_PATH}
fi
export CPATH=${JUNO_EXTLIB_libcurl_HOME}/include:${CPATH}
export MANPATH=${JUNO_EXTLIB_libcurl_HOME}/share/man:${MANPATH}

# For CMake search path
export CMAKE_PREFIX_PATH=${JUNO_EXTLIB_libcurl_HOME}:${CMAKE_PREFIX_PATH}

EOS

}


libcurl-wipe()
{
   libcurl-wipe-build
   libcurl-wipe-prefix
}

libcurl-wipe-build()
{
   local dir=$(libcurl-dir)
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

libcurl-wipe-prefix()
{
   local prefix=$(libcurl-prefix)
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

libcurl-find()
{
   find $(libcurl-prefix) -type f
}



