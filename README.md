# Spryped

> An Impedance Controlled Bipedal Robot

## Table of Contents

- [Setup](#setup)
- [Info](#info)

---

## Setup

1. Clone this directory wherever you want.

```shell 
git clone https://github.com/LocknutBushing/spryped.git
```  

2. Make sure both Python 3.7 and pip are installed.

```shell
sudo apt install python3.7
sudo apt-get install python3-pip
python3.7 -m pip install --upgrade pip
```

2. I recommend setting up a virtual environment for this, as it requires the use of a number of specific Python packages. The venv folder is already in the spryped directory, so after installing venv you just need to activate it.

```shell
sudo apt-get install python3.7-venv
cd spryped
source env/bin/activate
```
For more information on virtual environments: https://docs.python.org/3/library/venv.html
    
3. Now that the virtual environment is activated, install numpy, scipy, matplotlib, transforms3d, pyquaternion, PyBullet, and CasADi.

```shell
python3.7 -m pip install numpy scipy matplotlib transforms3d pyquaternion pybullet casadi
```
Don't use sudo here if you can help it, because it may modify your path and install the packages outside of the venv.

Here is the [PyBullet tutorial](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3).

---

## Info

This is the Github repo for a bipedal robot that I am developing.

To run the code:

```shell
python3.7 run.py
```
