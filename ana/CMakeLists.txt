cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name Ana)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

# order of increasing dependency
set(OLD_SCRIPTS

   __init__.py
   nbase.py
   enum_.py
   base.py
   debug.py
   log.py
   level.py
   qdv.py
   ctx.py
   make_rst_table.py
   abstat.py
   cfh.py
   dat.py
   bpath.py
   key.py
   env.py
   OpticksQuery.py
   main.py
   proplib.py
   decompression.py
   histype.py
   mattype.py
   hismask.py
   datedfolder.py
   metadata.py
   nibble.py
   evt.py
   profile_.py
   absmry.py
   ab.py
   tboolean.py

)


set(SCRIPTS
   __init__.py
   nbase.py
   base.py
   npmeta.py
   axes.py
   enum_.py
   qcf.py
   fold.py
   eget.py
   pvplt.py

   histype.py
   hismask.py
   nibble.py
   p.py
   seq.py
   nload.py
   num.py
)


install(PROGRAMS ${PARENT_DIR}/__init__.py  DESTINATION py/opticks)
install(PROGRAMS ${SCRIPTS}                 DESTINATION py/opticks/ana)

add_subdirectory(tests)

