#!/bin/bash -l 
msg="=== $BASH_SOURCE :"
name=UsePlogJsonStandalone
srcdir=$PWD
tmpdir=/tmp/$USER/opticks/$name


purl=https://github.com/simoncblyth/plog.git
#purl=https://github.com/SergiusTheBest/plog
PURL=${PURL:-$purl}
echo $msg PURL $PURL 

jurl=https://github.com/nlohmann/json/releases/download/v3.9.1/json.hpp
JURL=${JURL:-$jurl}
echo $msg JURL $JURL 


mkdir -p $tmpdir
cd $tmpdir
pwd

if [ -d "plog" ]; then
    echo $msg plog has already been cloned into $tmpdir/plog 
    echo $msg from URL : $(grep url $tmpdir/plog/.git/config) 
    echo $msg delete that plog folder to clone again from different URL
else
    echo $msg cloning from PURL $PURL   
    git clone $PURL
fi 

jsonpath=json/include/json.hpp
if [ -f "$jsonpath" ]; then
    echo $msg json has already been curled to $tmpdir/$jsonpath 
    echo $msg delete that file to curl again 
else
    mkdir -p $(dirname $jsonpath)
    curl -L $JURL -o $jsonpath
fi 




cat << EOS > $name.cc

#include <cstdlib>
#include "json.hpp"
#include <iostream>
#include <plog/Log.h> 

using json = nlohmann::json;

int main()
{
    const char* path = getenv("PLOGPATH") ; 
    std::cout << "writing to PLOGPATH " << path << std::endl ; 
    
    json j = { 
      {"pi", 3.141},
      {"happy", true},
      {"name", "Niels"},
      {"nothing", nullptr},
      {"answer", {
        {"everything", 42} 
      }}, 
      {"list", {1, 0, 2}},
      {"object", {
        {"currency", "USD"},
        {"value", 42.99}
      }}  
    };  

    j["extra"] =  {1,2,3} ; 

    std::cout << j << std::endl ; 

    plog::init(plog::debug, path ); 

    LOGD << "LOGD " ; 
    LOGD << j.dump(4) ;

    LOG_DEBUG << "LOG_DEBUG " ; 
    LOG_DEBUG << j.dump(4) ; 

    LOG(plog::debug) << "LOG(plog::debug) "  ; 
    LOG(plog::debug) << j.dump(4) ;  

    return 0;
}

EOS


echo $msg compile with c++11
gcc $name.cc -std=c++11 -lstdc++ -I$tmpdir/plog/include -I$tmpdir/json/include -o $tmpdir/$name 
[ $? -ne 0 ] && echo $msg compilation error with c++11 && exit 1 

if [ "$(uname)" == "Linux" ]; then 
    echo $msg compile with c++17
    gcc $name.cc -std=c++17 -lstdc++ -I$tmpdir/plog/include -I$tmpdir/json/include -o $tmpdir/$name 
    [ $? -ne 0 ] && echo $msg compilation error with c++17 && exit 1 
fi


export PLOGPATH=$tmpdir/$name.log 
$tmpdir/$name
[ $? -ne 0 ] && echo $msg run error && exit 2

echo $msg cat $PLOGPATH
cat $PLOGPATH

exit 0 
