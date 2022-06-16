# raadpy | RAAD Data Analysis Framework

This is a python package with the necessary libraries to conduct an analysis on the raw data obtained from the RAAD detector onboard the Light-1 cubesat mission.

# Table of contents

- [raadpy | RAAD Data Analysis Framework](#raadpy--raad-data-analysis-framework)
- [Table of contents](#table-of-contents)
- [Features](#features)
- [Installation](#installation)
  - [Installing with ``pip``](#installing-with-pip)
  - [Building from Source](#building-from-source)
    - [Download the code](#download-the-code)
    - [Make sure setuptools are up-to-date](#make-sure-setuptools-are-up-to-date)
    - [Build and install the ``raadpy``](#build-and-install-the-raadpy)
- [Basic Usage](#basic-usage)
  - [Load arrays of events from filenames](#load-arrays-of-events-from-filenames)
  - [Plot events on interactive maps](#plot-events-on-interactive-maps)


# Features

Here are some of the things you can do with ``raadpy``. Clicking the link will take you to tutorials on how to do any of these tasks.

1. [Load arrays of events from filenames](#load-arrays-of-events-from-filenames): ``raadpy`` can load different types of ecents, from lightning strikes to TGF events, to locations of the satellite, etc. Basically there is built-in support for everything that has longitude, latitude, and a timestamp. These arrays have extra features, such as automatic precision in storing timestamps and easy to use conversion between timestamp formats.
2. [Plot events on interactive maps](): After loading types of events one can plot them on interactive globes that can be exported as animations, interactive html files, or simply publication quality plots
3. [Automatically obtain lighting strikes near events](): Given a set of events (such as TGF events) ``raadpy`` can automatically detect nearby lightnings and download them in a python friendly format for computation.
4. [One-line reading of the Light-1 payload buffers](): the package can be used to decode the binary files from the buffers with 1 line of python code.
5. [Easy timestamp correction](): We all know what happened with the timestamps and the PPS signal. ``raadpy`` offers a simple way to estimate the timestamp using the order of the data in the payload buffers. 

These are only some things that the library can do, for a full list of the functions and capabilities please look at the [source code](https://github.com/nyuad-astroparticle/raadpy/tree/main/src/raadpy).

----
# Installation

``raadpy`` is a library build for ``python>=3.6`` however tested on ``python>=3.9``, we recommend updating to the latest python distribution before installing the code. We further recommend to clone [this python environment](https://github.com/nyuad-astroparticle/raad/tree/main/Conda_Environment_Installation) to download additional packages that would be useful. This step is not necessary however.

To install ``raadpy`` there are two options, installing the latest release through ``pip``, or installing from source. 

## Installing with ``pip``

To install thorugh **PyPI** using ``pip`` open a terminal and run

```terminal
$ pip install raadpy
```

This should install the latest version of ``raadpy`` automatically. In case this doesn't work, fear not! You can install the library from source.

## Building from Source

To build ``raadpy`` from source the following tools are needed:

   1. [**Git**](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git): Is a tool to download this code to your computer from the terminal. Link points to an installation tutorial.
   2. [**pip**](https://pypi.org/project/pip/): This is PyPI's package manager.

If the above is installed on your system then proceed with the following:

### Download the code

Open a terminal (Linux and MacOS), or a Powershell (Windows), and use the ``cd`` and ``ls`` (``dir`` in Windows) commands to navigate to the folder you want to download ``raadpy``'s source code. You can download it by running the following

```terminal
$ git clone https://github.com/nyuad-astroparticle/raadpy.git
$ cd raadpy/
```

After that you should have a directory called ``raadpy`` that contains the contents of this repo, and with the second command you should be in it.

### Make sure setuptools are up-to-date

To ensure that the tools you need to build the library are up to date run this next

``` terminal 
$ python3 -m pip install --upgrade pip setuptools wheel build
```

### Build and install the ``raadpy``

Next we will build ``raadpy`` by running

```terminal
$ python3 -m build
```

and we will install by running 

```terminal
$ pip install .
```

The "." at the end is important as it specifies that we want to install whateevr is in the current directory. And *voila!* You should have ``raadpy`` successfully installed.

---
# Basic Usage

This is a short tutorial to use ``raadpy``. It is meant as a short guide to understand how the package is structured and what are the main modules one can work with.

## Load arrays of events from filenames

With ``raadpy`` you can load events and convert them to a powerful python friendly format simply by using the filename. In this example we will load a set of TGF's from teh *Fermi* mission and then we will examine what we can do with this dataset.

You can do this in a jupyer notebook or in a simple python script. To load the data we will first import ``raadpy`` and then do the following

```python
# Import library
import raadpy as rp

# Define a variable with the filename where the FERMI data is stored
fermi_filename = "PATH-TO/FERMI-data.txt"     # Replace this path with yours

# Load the FERMI data
data = rp.array(filename=fermi_filename,event_type='fermi')
```

And that's it! Now you have created a ``raadpy array`` that contains the data from this filename specially formatted as *Fermi* mission data. 

A ``raadpy array`` can be thought of a list, it's structure is similar to a ``numpy array``, so you can still access an event by doing ``data[3]``, and perform all the other methods familliar to lists such as ``len(data)``, ``for datum in data``, etc. However this array has a special constructor that allows it to automatically format certain types of event. You can specify the type of event using the ``event_type`` argument as shown above. The following ``event_types`` are supported

1. ``location``: *(default)* Simply holds a location and timestamp of events over a map.
2. ``fermi``: TGFs from the fermi mission
3. ``light-1``: Events from the *Light-1* mission
4. ``lightning``: Lightnings usually downloaded from [blitzortung.org](https://www.blitzortung.org/en/live_lightning_maps.php).

You can set up your array to be any of these types, just know that ``event_type`` field is case sensitive, so entering ``LIght-1`` might result in an error.

But what can you do with the events after they are loaded? Well a bunch of things! First lets **print** a snapshot of them in order to examine them. This can simply be done by running

```python
# Print the data 
print(data)
```

this produces the follwoing output

```shell
0 TGF: TGF210628068  | Mission: fermi
Timestamp (ISO): 2021-06-28 01:37:54.440
Lat:   13.0000 	 Long:  -87.6833
Detector_id: 100010000000

1 TGF: TGF210627854  | Mission: fermi
Timestamp (ISO): 2021-06-27 20:30:00.815
Lat:   22.6000 	 Long: -100.2500
Detector_id: 101001

2 TGF: TGF210620681  | Mission: fermi
Timestamp (ISO): 2021-06-20 16:20:45.079
Lat:   -3.0167 	 Long:  144.0500
Detector_id: 10010000000010

3 TGF: TGF210617554  | Mission: fermi
Timestamp (ISO): 2021-06-17 13:17:19.041
Lat:   20.2667 	 Long:   78.1833
Detector_id: 11

4 TGF: TGF210617308  | Mission: fermi
Timestamp (ISO): 2021-06-17 07:24:08.879
Lat:   10.2500 	 Long:  -84.0333
Detector_id: 11

...
Lat:    2.7830 	 Long:  -69.9670
Detector_id: 11
```

As you can see the pkey of the event is printed, followed by it's unique ID, the mission time, the timestamp in ISO, then the latitude and longitude, and finally the id of the detector the event was from. 

Let's start using the library now, we can obtain a list of the timestamps of all of the envets in any time format like so:

```python
# Get the timestamps in mjd format
timestamps_mjd = data.get_timestamps(format = 'mjd')
print(timestamps_mjd)
```

the output is the following

```
array([59393.0679912, 59392.8541761, 59385.6810773, 59382.5536926,
       59382.3084361, 59379.7080586, 59371.2641016, 59354.7158131,
       59349.5049853, 59348.0465583, 59347.394978 , 59345.8690265,
       59339.3054766, 59338.8064969, 59338.3140316, 59337.4563384,
       59332.6334789, 59331.3091693, 59330.9530484, 59330.4564704,
...
       55635.6609862, 55632.1763637, 55630.7547783, 55629.2279641,
       55614.4684369, 55610.83396  , 55604.8978094], dtype=float64)
```

This actually offers arbitrary conversions to well known timestamps, here we chose ``mjd`` but you can use anything from ``unix``, ``iso``, and more! A numpy array is returned for easy further manipulation.

Similarly one can obtain tuples with the latitude and logitude of the elements of the arrays like so:

```python
# Get the coordinates of the events in lon-lat
positions = data.get_coords()
print(positions)
```

The output is as follows

```
array([[ -87.6833,   13.    ],
       [-100.25  ,   22.6   ],
       [ 144.05  ,   -3.0167],
       ...,
       [ 143.817 ,  -20.833 ],
       [ 100.667 ,    3.667 ],
       [ -69.967 ,    2.783 ]])
```

still a numpy array is returned with the appropriate entries as described above. 

## Plot events on interactive maps

With ``raadpy`` you can plot ``arrays`` in interactive maps using [``plotly``](https://plotly.com/). In this example we will load the path of the cubesat and plot it on an interactive javascript map.

First we load the data
```python
# Import the library
import raadpy as rp

# Filename of data
path_filename = 'PATH-TO/LIGHT-1_LOCATIONS.csv'  # Change the path accordingly

# Create raadpy array
data = rp.array(filename=path_filename,event_type='locations')

# Plot the first 10000 events
path = rp.array(data[:10000])   # Get the first 10000 events
rp.map(path,long=-80,lat=20)    # Plot them
```

The output looks like this

![plotly-earth](https://user-images.githubusercontent.com/31447975/174152009-2bc3573a-a54e-4ff5-b393-c9a84b4c441f.png)
