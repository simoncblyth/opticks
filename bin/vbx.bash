vbx-source(){ echo $BASH_SOURCE ; }
vbx-vi(){ vi $(vbx-source)  ; }
vbx-env(){  olocal- ; opticks- ; }
vbx-usage(){ cat << EOU

vbx : Virtual Box for testing Opticks on other Linux distros ?
================================================================================================

https://www.virtualbox.org/wiki/Linux_Downloads

Users of Oracle Linux / RHEL can add  the Oracle Linux repo file to /etc/yum.repos.d/. 


Objective
-----------

User Elias is reporting runtime nnode related SEGV issues 
with Opticks on Ubuntu 16.0.4 
with gcc 5.4.0 (?) at least CMake says 
"The CXX compiler identification is GNU 5.4.0" 

It would be good for me to reproduce the problem in such a system


April 4, 2019
--------------

Succeeded to install VirtualBox-6.0 onto precision workstation with::

   vbx-get-repo
   vbx-install

Installation message::

   creating group 'vboxusers'. VM users must be member of that group!

After this "Applications > System Tools > Oracle VM VirtualBox" is available


Download Ubuntu ISO from https://www.ubuntu.com/desktop 
 ~/Downloads/ubuntu-18.04.2-desktop-amd64.iso


Virtualbox tips
~~~~~~~~~~~~~~~~~~

* There is no "host" button but the right control button "rctrl" takes its place
* host+C switches between scaled and windowed mode
* host+home to access virtualbox menus, which otherwise have somehow disappeared 


Getting Ubuntu 18.04 kitted out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.tecmint.com/install-virtualbox-guest-additions-in-ubuntu/

::

   sudo apt install gcc curl mercurial git 

   sudo apt update
   sudo apt upgrade

   sudo apt install build-essential dkms linux-headers-$(uname -r)


Guest Additions not showing up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    sudo apt install net-tools   # want to ssh into the virtualbox ubuntu 
    sudo apt install openssh-server


Network Advanced Port Forwarding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forward port 2222 of the host to port 22 in the virtualbox 
which has IP 10.0.2.15 (obtained with ifconfig in the Ubuntu vbx)

::

    Name           Protocol   Host IP       Host Port    Guest IP      Guest Port 
    Rule 1         TCP        127.0.0.1     2222         10.0.2.15     22












EOU
}


vbx-get-repo()
{
    local msg="=== $FUNCNAME :"
    cd /etc/yum.repos.d/
    local url=https://download.virtualbox.org/virtualbox/rpm/el/virtualbox.repo
    local repo=$(basename $url)

    [ -f "$repo" ] && echo $msg repo $repo exists already && return  

    local cmd="sudo curl -L -O $url" 
    echo $msg $cmd
    eval $cmd
}

vbx-install()
{
    local cmd="sudo yum install VirtualBox-6.0"
    echo $msg $cmd
    eval $cmd
}


