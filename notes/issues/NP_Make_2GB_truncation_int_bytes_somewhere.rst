NP_Make_2GB_truncation_int_bytes_somewhere
============================================


Fixed this by switch int => NP::INT std::int64_t in hundreds of places in NP.hh
---------------------------------------------------------------------------------------

::

    P[blyth@localhost np]$ ~/np/tests/NP_Make_test.sh
             BASH_SOURCE : /home/blyth/np/tests/NP_Make_test.sh
                    name : NP_Make_test
                   sauce : NP_Make_test.cc
                  script : NP_Make_test.py
                     tmp : /data/blyth/np
                     TMP : /data/blyth/np
                    FOLD : /data/blyth/np/NP_Make_test
                  defarg : info_build_run_ana
                     arg : info_build_run_ana
                    test : Large
                    TEST : Large
    NP_Make_test::Large
    Python 3.7.7 (default, May  7 2020, 21:25:33) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.
    apath:/data/blyth/np/NP_Make_test/Large.npy st.st_size:2105032832 
    -rw-rw-r--. 1 blyth blyth 2105032832 Dec  3 10:03 /data/blyth/np/NP_Make_test/Large.npy
    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)
    ~/np/tests/NP_Make_test.py in <module>
         27         os.system("ls -l %s" % apath )
         28 
    ---> 29         a = np.load(apath)
         30 
         31         print("a.shape %s " % str(a.shape))

    ~/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding)
        438             else:
        439                 return format.read_array(fid, allow_pickle=allow_pickle,
    --> 440                                          pickle_kwargs=pickle_kwargs)
        441         else:
        442             # Try a pickle

    ~/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/lib/format.py in read_array(fp, allow_pickle, pickle_kwargs)
        769             array = array.transpose()
        770         else:
    --> 771             array.shape = shape
        772 
        773     return array

    ValueError: cannot reshape array of size 526258176 into shape (100000000,16)
    > /home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/lib/format.py(771)read_array()
        769             array = array.transpose()
        770         else:
    --> 771             array.shape = shape
        772 
        773     return array

    ipdb> 



