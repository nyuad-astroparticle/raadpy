# raadpy | RAAD Data Analysis Framework

<p align="center">
	<img src="https://img.shields.io/badge/raadpy-Active-green?style=for-the-badge" alt="raadpy">
</p>

<p align="center">
	<img src="https://img.shields.io/pypi/format/raadpy?style=flat-square" alt="Format">
	<img src="https://img.shields.io/pypi/v/raadpy?style=flat-square" alt="PyPI Release">
	<img src="https://img.shields.io/pypi/pyversions/raadpy?style=flat-square" alt="PyPI - Python Version">
	<img src="https://img.shields.io/github/v/release/nyuad-astroparticle/raadpy?style=flat-square" alt="GitHub release (latest SemVer)">
	<img src="https://img.shields.io/github/license/nyuad-astroparticle/raadpy?style=flat-square" alt="GitHub">
</p>

<!-- 
![raadpy](https://img.shields.io/badge/raadpy-Active-green?style=for-the-badge)

![Format](https://img.shields.io/pypi/format/raadpy?style=flat-square) 
![PyPI Release](https://img.shields.io/pypi/v/raadpy?style=flat-square)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/raadpy?style=flat-square)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/nyuad-astroparticle/raadpy?style=flat-square)
![GitHub](https://img.shields.io/github/license/nyuad-astroparticle/raadpy?style=flat-square)
 -->

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
  - [Automatically obtain lighting strikes near events](#automatically-obtain-lighting-strikes-near-events)
  - [One-line reading of the Light-1 payload buffers](#one-line-reading-of-the-light-1-payload-buffers)
  - [Easy timestamp correction](#easy-timestamp-correction)


# Features

Here are some of the things you can do with ``raadpy``. Clicking the link will take you to tutorials on how to do any of these tasks.

1. [Load arrays of events from filenames](#load-arrays-of-events-from-filenames): ``raadpy`` can load different types of ecents, from lightning strikes to TGF events, to locations of the satellite, etc. Basically there is built-in support for everything that has longitude, latitude, and a timestamp. These arrays have extra features, such as automatic precision in storing timestamps and easy to use conversion between timestamp formats.
2. [Plot events on interactive maps](#plot-events-on-interactive-maps): After loading types of events one can plot them on interactive globes that can be exported as animations, interactive html files, or simply publication quality plots
3. [Automatically obtain lighting strikes near events](#automatically-obtain-lighting-strikes-near-events): Given a set of events (such as TGF events) ``raadpy`` can automatically detect nearby lightnings and download them in a python friendly format for computation.
4. [One-line reading of the Light-1 payload buffers](#one-line-reading-of-the-light-1-payload-buffers): the package can be used to decode the binary files from the buffers with 1 line of python code.
5. [Easy timestamp correction](#easy-timestamp-correction): We all know what happened with the timestamps and the PPS signal. ``raadpy`` offers a simple way to estimate the timestamp using the order of the data in the payload buffers. 

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

<img src="https://user-images.githubusercontent.com/31447975/174157124-390287c3-5efc-405d-85b5-1e5ab33709a3.gif" alt="globe" width="400"/>

But the only difference is that in the python environment this is actually a 3D model that you can move, export into an image, html interactive file, or even an animation!


## Automatically obtain lighting strikes near events

To automatically obtain lightning strikes near events one can simply load a set of events, just like the *Fermi* TGFs loaded in the first example, and with a couple more commands the nearest lightnings will show up. This is at the code below

```python
# Pick a TFG from the data in the first example
TGF = data[10]

# Find all the ligtnings within 10s of it
lights = rp.download_lightnings(TGF.timestamp,threshold=10)

# Filter those lightnings to find the ones that are closest in time from the TGF
near = rp.get_nearby_lightning(TGF,lightnings=lights,threshold=1)
print(near)

# Finally plot the outcome on the map
rp.map(TGF,near,name1='TGF',name2='Lightning')
```

The output should look something like this

```
Warning! Threshold: 10.000000 s, is too small to be detected by Blitzortung! Using threshold = 6 * 60 s instead.
Searching for Lightnings between:
	 start-time: 2021-06-28 01:31:54.440
	 end-time:   2021-06-28 01:43:54.440
Found Lightning data at: https://www.blitzortung.org/en/archive_data.php?stations_users=0&selected_numbers=*&end_date=1624838400&end_time=6234&start_date=1624838400&start_time=5514&rawdata_image=0&north=90&west=-180&east=180&south=-90&map=0&width_orig=640&width_result=640&agespan=60&frames=12&delay=100&last_delay=1000&show_result=1
Data Downloaded Successfully
    0 Lightning: -79.891498 | Mission: Blitzurtong
        Timestamp (ISO): 2021-06-28 01:37:54.938
        Lat:   14.9226 	 Long:  -79.8915
        Detector_id: Blitz
        
    1 Lightning: -90.06834 | Mission: Blitzurtong
        Timestamp (ISO): 2021-06-28 01:37:53.891
        Lat:   33.1204 	 Long:  -90.0683
        Detector_id: Blitz
        
    2 Lightning: -90.026664 | Mission: Blitzurtong
        Timestamp (ISO): 2021-06-28 01:37:53.892
        Lat:   33.2297 	 Long:  -90.0267
        Detector_id: Blitz
...
```

<img width="417" alt="Screen Shot 2022-06-16 at 22 17 01" src="https://user-images.githubusercontent.com/31447975/174157153-4dcc75e3-9dc7-41d5-894c-f62ce6d03255.png">

The warning is not a big deal as it is simply saying that the website the data is downloaded from can only get the lightnings at intervals of 5 minutes. Therefore it is rebsing the search of lightnings to a 5 minute interval between them. Then the nearest lightnings are printed, and finally an interactive map of the lightnings is shown.

## One-line reading of the Light-1 payload buffers

Perhaps one of the most useful features of the ``raadpy`` library is that it can read the buffers of the Light-1 Payload with one-line commands, as well as filter the events accordingly. 

In this example we load the data for the orbit buffer and filter for the PMT signal only, then plot.

```python
# Import the library
import raadpy as rp

# Define the filename of the data
filename = 'PATH-TO-ORBIT/data.dat'  # Change this for your data

# Load the orbit buffer and apply a filter for the PMT data only
data = rp.get_dict(filename,struct=rp.ORBIT_STRUCT,condition="data['id_bit'] == 1")

# Plot the decoded data
fig, axes = rp.plot_buffer(data)
```

Note that the ``get_dict`` function returns a dictionary. Each entry of the dictionary is a numpy array of all the measurements on the file, and the entries of the dictionary are decided by predifined constants, in this case ``rp.ORBIT_STRUCT``. As a resutl, to read a buffer, one has to specify the structure of the file in a dictionary using the ``struct`` argument. The dictionary looks like the example shown below for the orbit buffer.

```python
ORBIT_STRUCT    = {
    'timestamp'     : 32,
    'temperature'   : 8,
    'rate0'         : 12,
    'rate1'         : 12,
    'rate2'         : 12,
    'rate3'         : 12,
    'ratev'         : 8,
    'hv_level'      : 12,
    'veto_level'    : 12,
    'id_bit'        : 1,
    'pps_active'    : 1,
    'suspended'     : 1,
    'power_on'      : 1,
    'scenario'      : 4,
}
```

The library however, has three structures predefined these are as follows so that the user doesn't have to do it from scratch:

1. **Orbit Buffer**: ``rp.ORBIT_STRUCT``
2. **Veto Buffer**: ``rp.VETO_STRUCT``
3. **Non-Veto Buffer**: ``rp.NONVETO_STRUCT``

So these can be used directly out of the box. An in-depth application of reading all the buffers is shown [here](https://github.com/nyuad-astroparticle/raad/blob/main/Data_Analysis/Decoders/buffer_reader.ipynb).

The output of the previous code looks like this

![orbit](https://user-images.githubusercontent.com/31447975/174181428-3a06ce0a-7481-4e49-b5ca-0b097685c074.png)

> **_NOTE:_**  There are two different ways that the buffers are encoded. If one wants to decode the Buffers for the VETO and NONVETO they must include ``STUPID=True`` as an argument on ``rp.get_dict()`` to specify the decoding method.


## Easy timestamp correction

As we know the corruption of the timestamp signal poses a major issue for the analysis of the Light-1 data. The algorithm derived to correct the timestamps is included in the library and can be invoked on a set of ``orbit`` data and ``buffer`` data like so

```python
# Filter events and correct timestamp based on orbit rate and buffer order
timestamp, total_cnt, valid_events = rp.correct_time(buffer,orbit,TIME,RANGE_ORBIT,RIZE_TIME,CONST_TIME,TMAX)
```

The arguments to ``rp.correct_time()`` are as follows:
   
1. ``buffer``: The decoded buffer that contains the events we are interested in correting their timestamp
2. ``orbit``: The equivalent orbit buffer
3. ``TIME``: The Period of the orbit data collection in seconds (normally set to 20s)
4. ``RANGE_ORBIT``: A tuple of indices that mark the beggining and end of the orbit data we want to look at (e.g. (300,5000))
5. ``RIZE_TIME``: The time it takes for the FPGA counter to reach saturation
6. ``CONST_TIME``: The time it the FPGA counter is saturated per cycle
7. ``TMAX``: The maximum number reached by the FPGA (999 for VETO and 9999 for NONVETO)

An example of using this function to correct the timestamp is shown [here](https://github.com/nyuad-astroparticle/raad/blob/main/Data_Analysis/Timestamp/rate.ipynb).
