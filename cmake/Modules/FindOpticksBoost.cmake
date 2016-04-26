
find_package(Boost REQUIRED COMPONENTS system thread program_options log log_setup filesystem regex)

if(Boost_FOUND)
set(OpticksBoost_FOUND ON)
set(OpticksBoost_LIBRARIES    ${Boost_LIBRARIES})
set(OpticksBoost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
set(OpticksBoost_DEFINITIONS  ${Boost_DEFINITIONS} -DBOOST_LOG_DYN_LINK)
endif(Boost_FOUND)


