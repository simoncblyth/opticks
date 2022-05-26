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



def copyref( l, g, s, kps ):
    """
    Copy selected references between scopes::
        
        copyref( locals(), globals(), self, "bnd,ubnd" )

    :param l: locals() 
    :param g: globals() or None
    :param s: self or None
    :param kps: space delimited string identifying quantities to be copied

    The advantage with using this is that can benefit from developing 
    fresh code directly into classes whilst broadcasting the locals from the 
    classes into globals for easy debugging. 
    """
    for k,v in l.items():
        kmatch = np.any(np.array([k.startswith(kp) for kp in kps.split()]))
        if kmatch:
            if not g is None: g[k] = v
            if not s is None: setattr(s, k, v )
            print(k)
        pass
    pass


def fromself( l, s, kk ):
    # nope seems cannot update locals() like this
    # possibly generate some simple code and eval it is a workaround 
    for k in kk.split(): 
        log.info(k)
        l[k] = getattr(s, k)
    pass


def shorten_bname(bname):
    elem = bname.split("/")
    if len(elem) == 4:
        omat,osur,isur,imat = elem
        bn = "/".join([omat,osur[:3],isur[:3],imat])
    else:
        bn = bname
    pass 
    return bn



