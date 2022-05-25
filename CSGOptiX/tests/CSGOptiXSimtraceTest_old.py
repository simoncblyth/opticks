    CSGOptiXSimtraceTest_OUTPUT_DIR = os.environ.get("CSGOptiXSimtraceTest_OUTPUT_DIR", None) 
    if CSGOptiXSimtraceTest_OUTPUT_DIR is None:
        log.fatal(" missing required envvar CSGOptiXSimtraceTest_OUTPUT_DIR ")
        sys.exit(1)
    pass

    CSGFoundry_DIR = CSGFoundry.FindDirUpTree( CSGOptiXSimtraceTest_OUTPUT_DIR, "CSGFoundry" )

    FOLD = os.path.dirname(CSGFoundry_DIR)

    LEAF = os.environ.get("LEAF", None)   ## LEAF is used to hop between geometries in sibling dirs 


    print( " CSGOptiXSimtraceTest_OUTPUT_DIR : %s " % CSGOptiXSimtraceTest_OUTPUT_DIR )
    print( " LEAF                            : %s " % LEAF )
    print( " CSGFoundry_DIR                  : %s " % CSGFoundry_DIR  )
    print( " FOLD                            : %s " % FOLD  )

    outdir = CSGOptiXSimtraceTest_OUTPUT_DIR 
    outbase = os.path.dirname(outdir)
    outleaf = os.path.basename(outdir)
    outgeom = outleaf.split("_")[0]  

    outdir2 = os.path.join(outbase, outleaf)  
    assert outdir == outdir2 
    print( " outleaf                         : %s " % outleaf )

    leaves = list(filter(lambda _:_.startswith(outleaf[:10]),os.listdir(outbase)))
    print("\n".join(leaves))

    if not LEAF is None and LEAF != outleaf:
        leafgeom = LEAF.split("_")[0]
        altdir = outdir.replace(outgeom,leafgeom).replace(outleaf,LEAF)
        if os.path.isdir(altdir):
            outdir = altdir
            print(" OVERRIDE CSGOptiXSimtraceTest_OUTPUT_DIR VIA LEAF envvar %s " % LEAF )
            print( " CSGOptiXSimtraceTest_OUTPUT_DIR : %s " % outdir )
        else:
            print("FAILED to override CSGOptiXSimtraceTest_OUTPUT_DIR VIA LEAF envvar %s " % LEAF )
        pass
    pass


