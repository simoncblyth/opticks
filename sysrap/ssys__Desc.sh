#!/bin/bash -l 

ssys__Desc()
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
ssys::Desc
-------------

Generated with::

   ~/opticks/sysrap/ssys__Desc.sh 
   ~/opticks/sysrap/ssys__Desc.sh | pbcopy 

Dump flags with::

    ssys_test

**/
inline std::string ssys::Desc()  // static
{
    std::stringstream ss ; 
    ss << "ssys::Desc"
       << std::endl 
EOH
ssys__Desc $(cat << EOB
CONFIG_Release
CONFIG_RelWithDebInfo
CONFIG_Debug
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

