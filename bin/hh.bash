hh-source(){ echo $BASH_SOURCE ; }
hh-vi(){ vi $(hh-source)  ; }
hh-env(){  olocal- ; opticks- ; }
hh-usage(){ cat << EOU

hh : Building Sphinx based documention using RST docstrings extracted from Opticks C++ headers
================================================================================================

hh.py 
     parses the docstrings from source headers and writes a 
     parallel tree of docstring .rst and index.rst to join 
     them together

hh--


Deficiencies
---------------

* order of projects should be in dependency order (or reverse), not current "random" ordering
* order of classes within subproj should be alphabetical
* Sphinx index is empty, some autogeneration ? 
* back navigation is missing
* many headers have no docstring
* not much use of :doc: linkage between pages 
* links across to bitbucket ?
* what about including notes too ?

Too much RST
-------------

I tend to write notes continuously while working on issues, these
are mostly devnotes of interest only to me (or perhaps developers)
and only for a short while after writing them.  

There are too many notes/issues in one directory

* thats real convenient for grepping to find issues that have resurfaced
* dont want to split them up at source level, but could do at 
  generated html level  

Generally I only want to look at notes that were edited recently.  
So need to make a selection of notes according to criteria.
Also different categories of RST need to be split up into 
separate trees for users and for developers. 

Types of RST 
-------------

* "live" issues from notes/issues with timestamp within the past 3 months for example
* "dead" issues from notes/issues with timestamp before cutoff
* "other" notes outside "notes/issues"
* "docstring" extracted from headers
* "manual" from Opticks source tree, excluding those from notes


Alt Tree
------------

Note that the tree of manually created .rst in the Opticks 
source tree is currently distinct from this experimental 
tree of RST docstrings extracted from the headers 
and assembled in a temporary dir such as $TMP/hh


Setup of Sphinx docs generated from Opticks header "docstrings"
--------------------------------------------------------------------

::

    cd $TMP/hh

    [blyth@localhost hh]$ which sphinx-quickstart
    ~/anaconda2/bin/sphinx-quickstart

    sphinx-quickstart
         ## answer the questions to generate a conf.py 

    make html 
         ## build the docstring pages

    open file:///tmp/blyth/opticks/hh/_build/html/index.html
         ## open in browser      
        



EOU
}

hh-dir(){   echo  $TMP/hh ; }
hh-cd(){ cd $(hh-dir) ; }
hh-find(){  find . -name '*.hh' -exec grep -l "/\*\*" {} \; ; }
hh-edit(){  vi $(opticks-home)/bin/hh.py ; }

hh--(){
   local hh
   hh-find | while read hh ; do 
      local l=$(cat $hh | hh.py --stdin | wc -l)

      if [ $l -gt 30 ]; then 
          printf "%40s : %d \n" $hh $l
      fi
   done
}


