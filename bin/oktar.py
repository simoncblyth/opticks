#!/usr/bin/env python
"""
oktar.py
==========

opticks binary distribution tarballs
-------------------------------------

This is used by the okdist- functions for creating
opticks binary distribution tarballs. Usage when testing::

    cd /usr/local/opticks
    ~/opticks/bin/oktar.py \
          /tmp/tt/Opticks-0.0.1_alpha.tar create \
         --prefix Opticks-0.0.1_alpha/i386-10.13.6-gcc4.2.1-geant4_10_04_p02-dbg

Convention for common prefix of all items in the archive::

   Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/


.opticks cache tarballs for sharing GEOM, rngcache, precooked etc..
----------------------------------------------------------------------

::

    cd ~/.opticks
    ~/opticks/bin/oktar.py /tmp/zz/dot_opticks.tar create --prefix dot_opticks/v0 --mode CACHE
    ~/opticks/bin/oktar.py /tmp/zz/dot_opticks.tar dump

* two elem prefix name combining the basis junosw_opticks_release name
  with an additional elem for the version of the "shared cache",
  no need for arch/gcc/os names as all noarch : mostly .npy files

    junosw_opticks_release_name/v1


"""
from pathlib import Path
import os, logging, sys, tarfile, argparse, shutil, textwrap, time
import getpass
import grp
import pwd

# Resolve current runtime user and group metadata once
CURRENT_UID = os.getuid()
CURRENT_GID = os.getgid()
CURRENT_UNAME = getpass.getuser()
try:
    CURRENT_GNAME = grp.getgrgid(CURRENT_GID).gr_name
except KeyError:
    CURRENT_GNAME = CURRENT_UNAME


def get_uname_gname(uid, gid):
    try:
        uname = pwd.getpwuid(uid).pw_name
    except KeyError:
        uname = str(uid)  # Fallback to string representation if ID doesn't exist

    try:
        gname = grp.getgrgid(gid).gr_name
    except KeyError:
        gname = str(gid)  # Fallback to string representation if ID doesn't exist

    return uname, gname



log = logging.getLogger(__name__)

class OKTar(object):
    """
    BASES lists prefix relative directories in order to
    control the starting points for the recursive adding to the archive

    They allow lots of irrelevant directories from the externals
    to be excluded from the archive by carefully selecting just what
    is needed.
    """
    BINARY_BASES = filter(None,textwrap.dedent(r"""
    envset.sh
    ENV.bash
    bashrc
    metadata
    bin
    lib
    lib64
    optix
    include
    cmake
    py
    gl
    tests
    externals/include/nljson
    #externals/include/ImGui
    #externals/include/GL
    #externals/include/GLFW
    #externals/imgui/imgui
    externals/imgui/imgui/extra_fonts
    externals/lib
    externals/lib64
    externals/plog/include
    externals/glm/glm/glm
    externals/share/bcm
    """).split("\n"))

    CACHE_BASES = filter(None,textwrap.dedent(r"""
    GEOM
    InputPhotons
    rngcache
    precooked
    flight
    """).split("\n"))

    @classmethod
    def Bases(cls, mode):
       assert mode in ["BINARY", "CACHE"], (mode,)
       BASES = cls.BINARY_BASES if mode == "BINARY" else cls.CACHE_BASES
       return list(map(str.strip,list(filter(lambda _:not _[0] == "#", BASES))))

    @classmethod
    def TarMode(cls, name):
        tarmode = None
        if name.endswith(".tar.gz"):
            tarmode = "w:gz"
        elif name.endswith(".tar"):
            tarmode = "w"
        pass
        assert not tarmode is None, "expecting name ending .tar.gz or .tar : not %s " % (name)
        return tarmode

    @classmethod
    def GetSize(self, path):
        if os.path.islink(path):
            sz = 0
        else:
            st = os.stat(path)
            sz = st.st_size
        pass
        return sz

    def __init__(self, path):
        """
        :param path: to the tarfile to create, extract from or dump
        """
        self.path = os.path.expanduser(path)
        self.name = os.path.basename(path)

        self.created_dirs = set()
        self.t = None
        self.pfx = "?"
        self.sztot = 0
        self.sz = {}

        self.exclude_dir_names = [".git",]

    def dump(self):
        self.t = tarfile.open(self.path, "r")
        for mi in self.t.getmembers():
            sz = mi.size/1e6
            if sz < 1.: continue
            print(" %10.3f : %s " % ( sz, mi.name))
        pass

    def create(self, prefix, mode):
        """
        :param prefix: relative root for all paths added to archive
        :param mode: either "BINARY" or "CACHE"
        """
        pass
        assert mode in ["BINARY", "CACHE"]

        tarmode = self.TarMode(self.name) # just looks at filename suffix
        base = os.path.realpath(os.getcwd())  # invoking directory

        outdir = os.path.dirname(self.path)  # directory of the archive
        if not os.path.isdir(outdir):
            log.info("creating outdir %s " % outdir)
            os.makedirs(outdir)
        pass
        log.info("prefix %s base %s " % (prefix, base))
        log.info("writing path %s tarmode %s mode %s " % (self.path, tarmode, mode) )

        self.prefix = prefix
        self.t = tarfile.open(self.path, tarmode)

        for name in self.Bases(mode):
            path = os.path.join(base, name)
            if not os.path.exists(path): continue
            if os.path.isfile(path):  ## top level files such as bashrc
                log.debug("adding top level file %s " % path)
                self.add_file(name)
            else:
                log.debug("adding top level directory %s " % path)
                self.recurse_(name, 0)
            pass
        pass
        self.add_toplink(".", "Opticks-vLatest")



    def recurse_(self, relbase, depth ):
        """
        :param relbase: base relative path eg starting with "lib", "include" etc..

        Recurse the tree calling self.add for all paths that
        are not excluded.
        """
        assert os.path.isdir(relbase), "expected directory %s does not exist " % relbase
        log.debug("relbase %s depth %s " % (relbase, depth))

        names = os.listdir(relbase)
        for name in names:
            relpath = os.path.join(relbase, name)
            if os.path.isdir(relpath):
                exclude = name in self.exclude_dir_names
                if not exclude:
                    self.recurse_(relpath, depth+1)
                pass
            else:
                self.add_file(relpath)
            pass
        pass

    def ensure_parent_dirs(self, arcname):
        """
        Ensures all parent directories are explicitly written TO THE TAR STREAM
        BEFORE child files are added.
        """
        arc_parents = list(Path(arcname).parents)

        # Reverse to process top-most parent first (e.g., 'ok', then 'ok/releases', etc.)
        for parent in reversed(arc_parents):
            parent_str = os.fspath(parent)
            if parent_str in (".", "/") or parent_str in self.created_dirs:
                continue

            # Ensure trailing slash for directory entry
            dir_arcname = parent_str if parent_str.endswith("/") else parent_str + "/"

            dir_info = tarfile.TarInfo(name=dir_arcname)
            dir_info.type = tarfile.DIRTYPE
            dir_info.mode = 0o755
            dir_info.mtime = int(time.time())

            self.t.addfile(dir_info)
            self.created_dirs.add(parent_str)
        pass


    def ensure_parent_dirs(self, arcname):
        """
        Ensures all parent directories are explicitly written TO THE TAR STREAM
        BEFORE child files are added.
        Uses filesystem metadata if the directory exists locally.
        """
        prefix_path = Path(self.prefix)
        arc_parents = list(Path(arcname).parents)

        # Reverse to process top-most parent first (top-down)
        for parent in reversed(arc_parents):
            parent_str = os.fspath(parent)
            if parent_str in (".", "/") or parent_str in self.created_dirs:
                continue

            # --- UNPREFIX TO RECOVER LOCAL FILESYSTEM PATH ---
            try:
                relpath = parent.relative_to(prefix_path)
            except ValueError:
                relpath = None

            # Check if directory actually exists on the filesystem
            fs_relpath = relpath if (relpath and relpath.exists()) else None
            dir_arcname = parent_str if parent_str.endswith("/") else parent_str + "/"

            dir_info = tarfile.TarInfo(name=dir_arcname)
            dir_info.type = tarfile.DIRTYPE

            if fs_relpath:
                st = fs_relpath.stat()

                uid, gid = st.st_uid, st.st_gid
                uname, gname = get_uname_gname(uid, gid)

                dir_info.uid = uid
                dir_info.gid = gid
                dir_info.uname = uname
                dir_info.gname = gname
                dir_info.mode = st.st_mode
                dir_info.mtime = int(st.st_mtime)
            else:
                # Fallback defaults for purely synthetic directories
                dir_info.mode = 0o755
                dir_info.mtime = int(time.time())
                dir_info.uid = CURRENT_UID
                dir_info.gid = CURRENT_GID
                dir_info.uname = CURRENT_UNAME
                dir_info.gname = CURRENT_GNAME
            pass

            self.t.addfile(dir_info)
            self.created_dirs.add(parent_str)




    def add_file(self, relpath):
        """
        :param relpath: real filesystem path relative to invoking directory
        """
        arcname = os.path.join(self.prefix,relpath)
        self.ensure_parent_dirs(arcname)

        self.t.add(relpath, arcname=arcname, recursive=False)

        sz = self.GetSize(relpath)
        self.sztot += sz
        self.sz[relpath] = sz

        if sz > 1e6:
            print(" %10.3f : %10.3f M : %s " % ( self.sztot/1e6, sz/1e6, relpath ))
        pass


    def add_toplink(self, relpath=".", linkname="Opticks-vLatest"):
        """
        :param relpath:
        :param linkname:

        Creates a sibling symlink inside the tar archive pointing to relpath.
        """
        arcname = Path(self.prefix) / relpath
        # Inject directory headers into tarball for the non-existent local parents
        self.ensure_parent_dirs(arcname)

        symlink_in_tar = os.fspath(arcname.with_name(linkname)) # link path inside archive
        link_target = arcname.name

        link_info = tarfile.TarInfo(name=symlink_in_tar)
        link_info.type = tarfile.SYMTYPE
        link_info.linkname = link_target
        # Set mtime from disk target (uses lstat to avoid following link if target is local link)
        link_info.mtime = int(os.path.getmtime(relpath))

        link_info.uid = CURRENT_UID
        link_info.gid = CURRENT_GID
        link_info.uname = CURRENT_UNAME
        link_info.gname = CURRENT_GNAME

        self.t.addfile(link_info)


    def extract(self, base):
        """
        :param base: directory in which to extract from archive

        Alternatively extract from commandline with eg::

            rm -rf Opticks-0.0.1_alpha
            tar xvf Opticks-0.0.1_alpha.tar

        In addition to that this checks the common prefix
        of the paths in the archive is following the two element
        convention.

        Note that because of the enforced use of a two level common prefix
        it is no problem to extract into the same directory as the creation,
        because this clears ahead.

        * Historically commonprefix was the input two level prefix : el9_amd64_gcc15_g411/Opticks-v0.6.7

        * But now that are adding symbolic link and full cvmfs base relative paths
          inside the tarball, the common file prefix is "ok/releases/el9_amd64_gcc15_g411/Opticks-v0.6.7"::

            [lo] A[blyth@localhost opticks_Debug_g411]$ tar tvf /data1/blyth/local/opticks_Debug_g411/ok_releases_el9_amd64_gcc15_g411_Opticks_v0_6_7.tar | tail -4
            -rw-r--r-- blyth/blyth     5952 2026-08-24 16:19 ok/releases/el9_amd64_gcc15_g411/Opticks-v0.6.7/externals/share/bcm/cmake/BCMProperties.cmake
            -rw-r--r-- blyth/blyth      367 2026-08-24 16:19 ok/releases/el9_amd64_gcc15_g411/Opticks-v0.6.7/externals/share/bcm/cmake/version.hpp
            -rw-r--r-- blyth/blyth     1406 2026-08-24 16:19 ok/releases/el9_amd64_gcc15_g411/Opticks-v0.6.7/externals/share/bcm/cmake/BCMConfig.cmake
            lrw-r--r-- blyth/blyth        0 2026-09-02 14:52 ok/releases/el9_amd64_gcc15_g411/Opticks-vLatest -> Opticks-v0.6.7


        """
        self.t = tarfile.open(self.path, "r")

        # Get names of regular files only (excludes directories, symlinks, etc.)
        file_names = [m.name for m in self.t.getmembers() if m.isfile()]

        # Calculate common path across files
        self.pfx = os.path.commonpath(file_names) if file_names else ""

        if not os.path.isdir(base):
            log.info("creating base %s " % base)
            os.makedirs(base)
        pass
        xdir = Path(base) / self.pfx

        log.info(f"commonpath prefix from file_names \"{self.pfx}\" base {base} xdir {xdir} ")

        if xdir.is_dir():
            log.info("common prefix extraction dir exists already %s " % xdir)
            log.info("removing xdir %s " % xdir )
            shutil.rmtree(xdir)
        pass
        log.info("extracting tarball with common prefix %s into base %s " % (self.pfx, base))
        if sys.version_info >= (3,12):
            self.t.extractall(base, filter='tar')
        else:
            self.t.extractall(base)
        pass

    def __str__(self):
        return "\n".join(self.n)
    def __repr__(self):
        return "OKTar %s %d " % ( self.pfx, len(self.n))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument( "path",  nargs=1, help="Path of distribution tarball, eg ~/Opticks-0.0.0_alpha.tar " )
    parser.add_argument( "verb", choices=["create","extract","dump"] )

    parser.add_argument( "--mode", choices=["BINARY", "CACHE" ], default="BINARY" )
    parser.add_argument( "--base",  default=os.getcwd(), help="Path at which to extract tarballs %(default)s ")
    parser.add_argument( "--prefix", default=None, help="sythetic prefix to all paths added to archive (ie relative to cvmfs root: /cvmfs/opticks.ihep.ac.cn/PREFIX)" )

    desc = { 'extract':"Extract tarball contents into base" ,
             'create':"Create with contents of current directory or --base argument if specified",
             'dump':"Dump names and sizes of tarball members" }

    parser.add_argument( "--level", default="info", help="logging level" )
    args = parser.parse_args()


    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)


    t = OKTar(args.path[0])

    if args.verb == "dump":
        t.dump()
    elif args.verb == "create":
        assert(args.prefix.startswith("ok/releases"))
        t.create(args.prefix, args.mode)
    elif args.verb == "extract":
        t.extract(args.base)
    pass



