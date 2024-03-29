#!/bin/bash -l 

QSim__Desc()
{
   local var 
   for v in $* ; do 
cat << EOF
#ifdef $v
       << "$v"
#else
       << "NOT:$v"
#endif
       << std::endl 
EOF
   done
}

cat << EOH
/**
QSim::Desc
------------

Generated with::

   ~/opticks/qudarap/QSim__Desc.sh 
   ~/opticks/qudarap/QSim__Desc.sh  | pbcopy 

Dump flags with::

   QSimDescTest 
   ssys__test

**/
std::string QSim::Desc()  // static
{
    std::stringstream ss ; 
    ss << "QSim::Desc"
       << std::endl 
EOH
# keep tag order same as ../sysrap/ssys__Desc.sh 
QSim__Desc $(cat << EOB
CONFIG_Debug
CONFIG_Release
CONFIG_RelWithDebInfo
CONFIG_MinSizeRel
PRODUCTION
WITH_CHILD
WITH_CUSTOM4
PLOG_LOCAL
DEBUG_PIDX
DEBUG_TAG
EOB
)

cat << EOT
       ;
    std::string str = ss.str() ; 
    return str ;  
}
EOT

