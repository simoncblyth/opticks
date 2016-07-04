brc-src(){      echo boostrapclient/brc.bash ; }
brc-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(brc-src)} ; }
brc-vi(){       vi $(brc-source) ; }
brc-usage(){ cat << \EOU

BoostRap Client Testing
==========================

Run in VS
------------

Load Solution into Visual Studio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   PS> vs-export
   PS> devenv C:\usr\local\opticks\build\boostrapclient\BoostRapClient.sln


Setup GUI
~~~~~~~~~~~~

* right click target "BoostRapClient"  and "Set As Startup Project"
* set Project > Properties > Debugging > Environment to::

   PATH=C:\usr\local\opticks\lib;%PATH%$(LocalDebuggerEnvironment)


F5 : Run in Debugger : Reveals Issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pops up a dialog (non-copy/pastable)::

   The procedure entry point GetSystemTimeAsFileTime could not 
   be located in the dynamic link library KERNEL32.dll.

::

    'BoostRapClient.exe' (Win32): Loaded 'C:\usr\local\opticks\build\boostrapclient\Debug\BoostRapClient.exe'. Symbols loaded.
    'BoostRapClient.exe' (Win32): Loaded 'C:\Windows\SysWOW64\ntdll.dll'. Cannot find or open the PDB file.
    'BoostRapClient.exe' (Win32): Loaded 'C:\Windows\SysWOW64\kernel32.dll'. Cannot find or open the PDB file.
    'BoostRapClient.exe' (Win32): Loaded 'C:\Windows\SysWOW64\KernelBase.dll'. Cannot find or open the PDB file.
    'BoostRapClient.exe' (Win32): Loaded 'C:\usr\local\opticks\lib\BoostRap.dll'. Symbols loaded.
    Exception thrown at 0x779618D6 (ntdll.dll) in BoostRapClient.exe: 0xC0000139: Entry Point Not Found.
    The program '[10748] BoostRapClient.exe' has exited with code -1073741511 (0xc0000139) 'Entry Point Not Found'.



Skip timeutil.cc from brap- allows brc to run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    PS C:\usr\local\opticks\lib> .\BoostRapClient.exe
     argc 1 argv[0] C:\usr\local\opticks\lib\BoostRapClient.exe
    BDemo::check i 42
    checked







EOU
}
brc-env(){      elocal- ; opticks- ; vs- ;  }

brc-dir(){  echo $(opticks-home)/boostrapclient ; }
brc-bdir(){ echo $(opticks-prefix)/build/boostrapclient ; }
brc-cd(){   cd $(brc-dir); }
brc-bcd(){  cd $(brc-bdir); }


brc-name(){ echo BoostRapClient ; }
brc-bexe(){ echo $(brc-bdir)/Debug/$(brc-name).exe ; }
brc-exe(){  echo $(opticks-prefix)/lib/$(brc-name).exe ; }
brc-sln(){  echo $(brc-bdir)/$(brc-name).sln ; }

brc-slnw(){ echo $(vs-wp $(brc-sln)) ; }


brc-run()
{
   PATH=$(opticks-prefix)/lib:$(opticks-prefix)/externals/lib:"$PATH" $(brc-exe)
}





