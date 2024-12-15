update_cvmfs_geometry_folders
==============================


History
--------

Sep 11, 2024
    inplace update /cvmfs/opticks.ihep.ac.cn/.opticks/GEOM/J_2024aug27 
    following fix to SMesh::serialize to include the lvid metadata needed
    for SScene->SScene sub-selection



Instructions
--------------


1. check currently configured GEOM with GEOM bash function
2. create tarball of the currently configured GEOM::

   ~/o/bin/GEOM_tar.sh 

   * if .tar exists already but an update is needed, delete the tar and rerun::

    [blyth@localhost sop]$ rm ~/.opticks/GEOM/J_2024aug27.tar
    [blyth@localhost sop]$ ~/opticks/bin/GEOM_tar.sh
                       0 : /home/blyth/opticks/bin/GEOM_tar.sh 
                    BASE : /home/blyth/.opticks/GEOM 
                    GEOM : J_2024aug27 
    GEOM J_2024aug27 is defined
    GEOM J_2024aug27 directory /home/blyth/.opticks/GEOM/J_2024aug27 exists
    GEOM /home/blyth/.opticks/GEOM/J_2024aug27.tar exists already
    499M    J_2024aug27.tar
    [blyth@localhost sop]$ 


3. scp the tarball to lxlogin::

   scp ~/.opticks/GEOM/J_2024aug27.tar L:g/   ## use g to avoid afs

    ## or use : ~/o/bin/GEOM_tar.sh scp 


4. copy from lxlogin to cvmfs stratum 0::

    ssh L

    [blyth@lxlogin004 ~]$ scp J_2024aug27.tar O:
    Enter passphrase for key '/afs/ihep.ac.cn/users/b/blyth/.ssh/id_rsa': 
    J_2024aug27.tar       

    scp J_2024nov27.tar O:


5. ssh from lxlogin to cvmfs stratum 0::

    [blyth@lxlogin004 ~]$ ssh O
    Enter passphrase for key '/afs/ihep.ac.cn/users/b/blyth/.ssh/id_rsa': 
    Last login: Thu Aug 29 21:10:31 2024 from lxlogin002.ihep.ac.cn
    [optickspub@cvmfs-stratum-zero-b ~]$

6. review instructions::

    [optickspub@cvmfs-stratum-zero-b ~]$ cat instructions.txt 

    To start a transaction::

        cvmfs_server transaction opticks.ihep.ac.cn

    To edit repo::

        cd /cvmfs/opticks.ihep.ac.cn 

        edit repo usually by copying arch labelled tarball in 

        cd /cvmfs   # step away from the directory being changed 

    To publish a transaction::

        cvmfs_server publish -m "your publish info" opticks.ihep.ac.cn


7. explode the new .tar ontop of existing directory tree within in transaction::


    cvmfs_server transaction opticks.ihep.ac.cn

    cd /cvmfs/opticks.ihep.ac.cn/.opticks/GEOM

    tar tvf ~/J_2024aug27.tar   ## check paths
    tar xvf ~/J_2024aug27.tar   ## explode ontop of existing directory tree

    cd /cvmfs ; cvmfs_server publish -m "inplace update /cvmfs/opticks.ihep.ac.cn/.opticks/GEOM/J_2024aug27 " opticks.ihep.ac.cn 
    cd /cvmfs ; cvmfs_server publish -m "add /cvmfs/opticks.ihep.ac.cn/.opticks/GEOM/J_2024nov27 " opticks.ihep.ac.cn 


::

    [optickspub@cvmfs-stratum-zero-b GEOM]$ cd /cvmfs ; cvmfs_server publish -m "inplace update /cvmfs/opticks.ihep.ac.cn/.opticks/GEOM/J_2024aug27 " opticks.ihep.ac.cn 
    Using auto tag 'generic-2024-09-03T07:32:35Z'
    WARNING: cannot apply pathspec /software/*/*
    WARNING: cannot apply pathspec /software/*/*/*
    Processing changes...
    .............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
    Waiting for upload of files before committing...
    Committing file catalogs...
    Note: Catalog at / gets defragmented (41.64% wasted row IDs)... done
    Wait for all uploads to finish
    Exporting repository manifest
    Statistics stored at: /var/spool/cvmfs/opticks.ihep.ac.cn/stats.db
    Tagging opticks.ihep.ac.cn
    Flushing file system buffers
    Signing new manifest
    Remounting newly created repository revision
    [optickspub@cvmfs-stratum-zero-b cvmfs]$ 





