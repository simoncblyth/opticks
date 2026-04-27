libcurl-env(){      opticks- ;  }
libcurl-vi(){       vi $BASH_SOURCE ; }
libcurl-usage(){ cat << EOU





EOU
}

libcurl-deps(){
   : check that all deps other than libz and libc are controlled
   local externals=/tmp/$USER/opticks/externals
   source $externals/openssl/openssl-3.2.0/bashrc
   source $externals/libcurl/curl-8.12.1/bashrc
   env | grep JUNO
   ldd $JUNO_EXTLIB_libcurl_HOME/lib64/libcurl.so
}


libcurl-info(){ cat << EOI

   libcurl-dir    : $(libcurl-dir)
   libcurl-bdir   : $(libcurl-bdir)
   libcurl-idir   : $(libcurl-idir)
   libcurl-prefix : $(libcurl-prefix)

EOI
}

libcurl-dir(){  echo $(opticks-prefix)/externals/libcurl/$(libcurl-name) ; }
libcurl-bdir(){ echo $(libcurl-dir).build ; }

#libcurl-idir(){ echo $(opticks-prefix)/externals ; }
libcurl-idir(){ echo /tmp/$USER/opticks/externals/libcurl/$(libcurl-name) ; }

libcurl-prefix(){ echo $(libcurl-idir) ; }

libcurl-cd(){  cd $(libcurl-dir); }
libcurl-bcd(){ cd $(libcurl-bdir); }
libcurl-icd(){ cd $(libcurl-idir); }

libcurl-version(){ echo 8.12.1 ; }
libcurl-name(){ echo curl-$(libcurl-version) ; }  # NB no lib
libcurl-url(){
   case $(libcurl-version) in
      8.12.1) echo https://curl.se/download/curl-8.12.1.tar.gz ;;
   esac
}


libcurl-get(){
   local dir=$(dirname $(libcurl-dir)) &&  mkdir -p $dir && cd $dir

   local rc=0
   local url=$(libcurl-url)
   local tgz=$(basename $url)
   local opt=$( [ -n "${VERBOSE}" ] && echo "-xzf" || echo "-xzvf" )

   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && opticks-curl $url
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
    -DBUILD_CURL_EXE=OFF \
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


libcurl--()
{
   local msg="=== $FUNCNAME :"
   libcurl-get
   [ $? -ne 0 ] && echo $msg get FAIL && return 1

   libcurl-build
   [ $? -ne 0 ] && echo $msg build FAIL && return 2

   libcurl-bashrc
   [ $? -ne 0 ] && echo $msg bashrc FAIL && return 2

   return 0
}


libcurl-bashrc()
{
   if [ "$(libcurl-prefix)" != "$(opticks-prefix)/externals" ]; then
       local path=$(libcurl-prefix)/bashrc
       echo $BASH_SOURCE $FUNCNAME - writing $path - prefix needs to be named and versioned for this to make sense
       libcurl-bashrc- > $path
   fi
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


libcurl-wipe(){
  local bdir=$(libcurl-bdir)
  rm -rf $bdir

  libcurl-wipe-manifest
}


libcurl-wipe-manifest()
{
    local prefix=$(libcurl-prefix)
    [ ! -d "$prefix" ] && echo "Prefix $prefix does not exist." && return

    echo "Wiping libcurl from $prefix..."

    # 1. Disable literal string treatment, allow globbing
    # 2. Iterate through the manifest
    libcurl-manifest | while read -r rel; do
        # Use an array to handle glob expansion
        local paths=( ${prefix}/${rel} )

        for path in "${paths[@]}"; do
            if [ -e "$path" ] || [ -L "$path" ]; then
                echo "Removing $path"
                rm -rf "$path"
            fi
        done
    done

    # Cleanup empty leaf directories
    find "$prefix" -type d -empty -delete 2>/dev/null
}


libcurl-manifest(){ cat << EOM
share/man/man3/curl*
share/man/man3/libcurl*
share/man/man3/CURL*
share/man/man1/curl-config.1
share/man/man1/mk-ca-bundle.1
lib64/libcurl.*
bin/curl-config
lib64/pkgconfig/libcurl.pc
include/curl
lib64/cmake/CURL
lib64/cmake/CURL
lib64/cmake/CURL
lib64/cmake/CURL
bin/mk-ca-bundle.pl
EOM
}







