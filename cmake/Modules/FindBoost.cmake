set(Boost_PREFIX "/opt/local")

find_library( Boost_system_LIBRARY 
              NAMES boost_system-mt
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_thread_LIBRARY 
              NAMES boost_thread-mt
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_program_options_LIBRARY 
              NAMES boost_program_options-mt
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_log_LIBRARY 
              NAMES boost_log-mt
              PATHS ${Boost_PREFIX}/lib )


set(Boost_LIBRARIES 
     ${Boost_system_LIBRARY}
     ${Boost_thread_LIBRARY}
     ${Boost_program_options_LIBRARY}
     ${Boost_log_LIBRARY}
)
set(Boost_INCLUDE_DIRS "${Boost_PREFIX}/include")
set(Boost_DEFINITIONS "-DBOOST_LOG_DYN_LINK" )

