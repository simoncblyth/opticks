SEvt_relative_save_directory_mode_would_be_useful
===================================================


With OPTICKS_MODE_SAVE:1 use relative approach at get them closer
--------------------------------------------------------------------

::

    A[blyth@localhost tests]$ l /tmp/blyth/opticks/oj_test/hitlitemerged/
    total 1032
    832 -rw-r--r--. 1 blyth blyth 731043 Dec  4 11:49 python3.11.log
    124 -rw-r--r--. 1 blyth blyth 125670 Dec  4 11:49 sample_detsim_user.root
     64 -rw-r--r--. 1 blyth blyth  62100 Dec  4 11:49 sample_detsim.root
      0 drwxr-xr-x. 3 blyth blyth    171 Dec  4 11:49 .
      0 drwxr-xr-x. 3 blyth blyth     18 Dec  4 11:49 ALL0_no_opticks_event_name
      4 -rw-r--r--. 1 blyth blyth   1576 Dec  4 11:49 run_meta.txt
      4 -rw-r--r--. 1 blyth blyth    132 Dec  4 11:49 run.npy
      4 -rw-r--r--. 1 blyth blyth   1734 Dec  4 11:49 SProf.txt
      0 drwxr-xr-x. 3 blyth blyth     27 Nov 21 14:51 ..
    A[blyth@localhost tests]$ l /tmp/blyth/opticks/oj_test/hitlitemerged/ALL0_no_opticks_event_name/
    total 4
    0 drwxr-xr-x. 3 blyth blyth  171 Dec  4 11:49 ..
    0 drwxr-xr-x. 3 blyth blyth   18 Dec  4 11:49 .
    4 drwxr-xr-x. 2 blyth blyth 4096 Dec  4 11:49 A000
    A[blyth@localhost tests]$ l /tmp/blyth/opticks/oj_test/hitlitemerged/ALL0_no_opticks_event_name/A000/
    total 536
      4 drwxr-xr-x. 2 blyth blyth   4096 Dec  4 11:49 .
      0 drwxr-xr-x. 3 blyth blyth     18 Dec  4 11:49 ..
     12 -rw-r--r--. 1 blyth blyth   8768 Dec  4 11:49 genstep.npy
     28 -rw-r--r--. 1 blyth blyth  25904 Dec  4 11:49 hitlitemerged.npy
      4 -rw-r--r--. 1 blyth blyth     81 Dec  4 11:49 NPFold_index.txt
      4 -rw-r--r--. 1 blyth blyth    673 Dec  4 11:49 NPFold_meta.txt
      0 -rw-r--r--. 1 blyth blyth      0 Dec  4 11:49 NPFold_names.txt
    144 -rw-r--r--. 1 blyth blyth 147248 Dec  4 11:49 photonlite.npy
     40 -rw-r--r--. 1 blyth blyth  36908 Dec  4 11:49 seqnib.npy
      4 -rw-r--r--. 1 blyth blyth    260 Dec  4 11:49 seqnib_table.npy
    288 -rw-r--r--. 1 blyth blyth 294368 Dec  4 11:49 seq.npy
      4 -rw-r--r--. 1 blyth blyth    192 Dec  4 11:49 sframe_meta.txt
      4 -rw-r--r--. 1 blyth blyth    384 Dec  4 11:49 sframe.npy
    A[blyth@localhost tests]$ 





With default absolute approach the outputs end in unconnected folders
-----------------------------------------------------------------------

::

    [blyth@localhost sysrap]$ l /tmp/blyth/opticks/oj_test/hitlitemerged/
    total 616
    416 -rw-r--r--. 1 blyth blyth 422304 Dec  4 10:48 python3.11.log
    124 -rw-r--r--. 1 blyth blyth 125670 Dec  4 10:48 sample_detsim_user.root
     64 -rw-r--r--. 1 blyth blyth  62075 Dec  4 10:48 sample_detsim.root
      0 drwxr-xr-x. 2 blyth blyth    137 Dec  4 10:48 .
      4 -rw-r--r--. 1 blyth blyth   1556 Dec  4 10:48 run_meta.txt
      4 -rw-r--r--. 1 blyth blyth    132 Dec  4 10:48 run.npy
      4 -rw-r--r--. 1 blyth blyth   1734 Dec  4 10:48 SProf.txt
      0 drwxr-xr-x. 3 blyth blyth     27 Nov 21 14:51 ..
    A[blyth@localhost sysrap]$ l /tmp/blyth/opticks/GEOM/J25_4_0_opticks_Debug/python3.11/ALL0_no_opticks_event_name/A000/
    total 564
     28 -rw-r--r--. 1 blyth blyth  25904 Dec  4 10:48 hitlitemerged.npy
      4 -rw-r--r--. 1 blyth blyth     81 Dec  4 10:48 NPFold_index.txt
      4 -rw-r--r--. 1 blyth blyth    673 Dec  4 10:48 NPFold_meta.txt
      0 -rw-r--r--. 1 blyth blyth      0 Dec  4 10:48 NPFold_names.txt
    144 -rw-r--r--. 1 blyth blyth 147248 Dec  4 10:48 photonlite.npy
     40 -rw-r--r--. 1 blyth blyth  36908 Dec  4 10:48 seqnib.npy
      4 -rw-r--r--. 1 blyth blyth    260 Dec  4 10:48 seqnib_table.npy
    288 -rw-r--r--. 1 blyth blyth 294368 Dec  4 10:48 seq.npy
      4 -rw-r--r--. 1 blyth blyth    192 Dec  4 10:48 sframe_meta.txt
      4 -rw-r--r--. 1 blyth blyth    384 Dec  4 10:48 sframe.npy
     12 -rw-r--r--. 1 blyth blyth   8768 Dec  4 10:48 genstep.npy
      4 drwxr-xr-x. 2 blyth blyth   4096 Nov 21 14:53 .
     28 -rw-r--r--. 1 blyth blyth  25952 Nov 14 22:39 hitlite.npy
      0 drwxr-xr-x. 3 blyth blyth     18 Nov 13 19:41 ..
    A[blyth@localhost sysrap]$ date
    Thu Dec  4 11:02:00 AM CST 2025
    A[blyth@localhost sysrap]$ 




::

    4356	void SEvt::save()
    4357	{
    4358	    const char* base = DefaultBase();
    4359	    LOG_IF(info, LIFECYCLE || SIMTRACE) << " base [" << ( base ? base : "-" ) << "]" ;
    4360	    save(base);
    4361	}
    4362	
    (gdb) b 4360
    Breakpoint 2 at 0x7fffbe3dc4fa: file /home/blyth/opticks/sysrap/SEvt.cc, line 4360.
    (gdb) c
    Continuing.

    Thread 1 "python" hit Breakpoint 2, SEvt::save (this=0xc5ca8e0) at /home/blyth/opticks/sysrap/SEvt.cc:4360
    4360	    save(base);
    (gdb) p base
    $1 = 0x7fffbe5075c8 "$TMP/GEOM/$GEOM/$ExecutableName"
    (gdb) 



