#############################
#       RAAD Library        #
#############################

# Import necessary Libraries
from operator import le
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import to_hex
import pandas as pd
import numpy as np
import geoviews as gv
import datetime as dt
import requests
from lxml import html
from gzip import decompress
from tqdm.notebook import tqdm
gv.extension('bokeh', 'matplotlib')

##################################################################################
# Useful Constants
##################################################################################
data_dir        = '../../Data/RAW/'        # Filename with directories
BYTE            = 8                        # Byte length
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

VETO_STRUCT     = {
    'channel'       : 2,
    'adc_counts'    : 14,
    'veto'          : 8,
    'stimestamp'    : 40, 
}

NONVETO_STRUCT  = {
    'channel'       : 2,
    'adc_counts'    : 10,
    'stimestamp'    : 36,
}

##################################################################################
# Helper functions
##################################################################################

# helper function to convert longitude in range -180 to 180
def in_range(longitude):
    return longitude if longitude <= 180 else longitude - 360


# Get an astropy time object, and return a string with the timestamp in epoch
def get_epoch_date(date_time):
    # Convert to datetime
    date = date_time.to_datetime()
    date = dt.datetime(date.year,date.month,date.day)

    # Return a string with just the date
    return str(int(Time(date).to_value('unix')))

# Get an astropy object, and retun a string with the time in epoch
def get_epoch_time(date_time):
    # Get just the date
    date = date_time.to_datetime()
    date = dt.datetime(date.year,date.month,date.day)

    # Subtract the date from the original datetime to get the time
    time = date_time.to_datetime() - date

    return str(time.seconds)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

##################################################################################
# Classes
##################################################################################

# TGF Class
class event: 
    # Constructor ###############################################################
    def __init__(self,timestamp,latitude:float,longitude:float,detector_id:str,event_id:str='',mission:str='',time_format:str='mjd',event_type:str='TGF'):
        # Set up the variables
        self.timestamp      = Time(timestamp, format=time_format)
        self.latitude       = latitude
        self.longitude      = longitude
        self.detector_id    = detector_id
        self.event_id       = event_id
        self.mission        = mission
        self.event_type     = event_type

    
    # SOME FUNCTIONS FOR THE TGF CLASS ##########################################

    # Return a string for that TGF
    def __str__(self):
        str = ''' %s: %s | Mission: %s
        Timestamp (ISO): %s
        Lat: %9.4f \t Long: %9.4f
        Detector_id: %s
        '''%(self.event_type,self.event_id,self.mission,self.get_iso(),self.latitude,self.longitude,self.detector_id)

        return str

    # Print a TGF
    def print(self):
       print(self.__str__())

    # Get any time format that astropy has to offer
    def get_timestamp(self,time_format:str='mjd',data_type:str='long'):
        data_type = None if time_format == 'iso' else data_type
        return self.timestamp.to_value(time_format,data_type)

    # Return EPOCH Date
    def get_epoch(self):
        return self.get_timestamp('unix')

    # Return MJD Date
    def get_mjd(self):
        return self.get_timestamp('mjd') 

    # Return ISO Date
    def get_iso(self):
        return self.get_timestamp('iso',None)
    

# Since we will be working with tgf arrays so much we will create
# Another class called TGF Array that has methods to handle an array of TGFs
class array:
    event_types = ['fermi','light-1','lightning']

    # Constructor
    def __init__(self,events=None,filename='',event_type=event_types[0]):
        
        # Add the tgfs as an array
        self.events = [] if events is None else events

        # if you add a filename, add the apppend the tgfs from that filename
        if filename != '':
            self.from_file(filename=filename,event_type=event_type,append=True)


    # Method to append
    def append(self,ev):
        self.events.append(ev)
    
    # Method to convert this into a string
    def __str__(self):
        string = ''
        for i,ev in enumerate(self.events):
            string += '%5d'%i + str(ev) + '\n'
    
        return string

    # Make objects compatible with len()
    def __len__(self):
        return len(self.events)

    # Overload the [] opeartor to be able to do tgf_array[1] etc.
    def __getitem__(self,i):
        if type(i) == list: return array([self.events[index] for index in i])
        else: return self.events[i]

    # Overload the [] operator to be able to do tgf_array[1] = 3 etc.
    def __set_item__(self,i,value):
        if type(i) == list:
            if type(value) == list:
                assert(len(value) == len(i))
                for index in i: self.events[index]=value[index]
            else:
                for index in i: self.events[index]=value
        
        else: self.events[i] = value

    # And a print method
    def print(self):
        print(self.__str__())

    
    # Get longitude and latitude as numpy arrays
    def get_coords(self):
        coords = []
        for event in self.events:
            coords.append([event.longitude,event.latitude])
        
        return np.array(coords)

    # Get array of timestamps
    def get_timestamps(self,format=None):
        times = []
        for event in self.events:
            if format is None:
                times.append(event.timestamp)
            else:
                times.append(event.get_timestamp(format))

        return np.array(times)

    # To list function
    def to_list(self):
        return self.events



    # Generate an array of tgfs from a file
    def fermi_from_file(self,filename,append:bool=True):
        # Load the TGF data
        data = pd.read_csv(filename).to_numpy()
        tgfs = []                                   # List to store the TGFs

        # For all the TGFs in the loaded dataset
        for datum in data:
            # Create a TGF and append it to the array
            tgfs.append(event(timestamp = datum[5],
                            longitude   = in_range(float(datum[9])), 
                            latitude    = float(datum[10]),
                            detector_id = datum[8],
                            event_id    = datum[2],
                            mission     = 'fermi',
                            time_format = 'mjd',
                            event_type  = 'TGF'))
        
        # If you want to append the data to the original array do so here
        if append: self.events += tgfs

        # Otherwise return them as a different tgf_array
        else: return array(tgfs)

    # Generate an array of lightnings from a file
    def lightning_from_file(self,filename:str,append:bool=True):
        # Load the lightning data
        data    = pd.read_csv(filename).to_numpy()
        lights  = []                                   # List to store the ligtnigs

        # For all the TGFs in the loaded dataset
        for datum in data:
            # Create a TGF and append it to the array
            lights.append(event(timestamp   = float(datum[0]) * 1e-9,
                                longitude   = in_range(float(datum[2])), 
                                latitude    = float(datum[1]),
                                detector_id = 'Blitz',
                                event_id    = datum[2],
                                mission     = 'Blitzurtong',
                                time_format = 'unix',
                                event_type  = 'Lightning'))
        
        # If you want to append the data to the original array do so here
        if append: self.events += lights

        # Otherwise return them as a different tgf_array
        else: return array(lights)
    
    def from_file(self,filename:str,event_type:str=event_types[0],append:bool=True):
        # Choose the appropriate function to load the data
        if   event_type == array.event_types[0]:
            return self.fermi_from_file(filename=filename, append=append)
        
        elif event_type == array.event_types[2]:
            return self.lightning_from_file(filename=filename, append=append)



##################################################################################
# Working functions
##################################################################################

# Visualize a set of TGFs on a map
def map(tgfs,lightnings:array=None):
    # If it is a single point, convert it into an array
    if type(tgfs)   == list:        tgfs = array(tgfs)
    elif type(tgfs) == event:       tgfs = array([tgfs])
    elif type(tgfs) != array: raise Exception("type %s is not an event object nor a list. Please enter a TGF object"%type(tgfs))

    # Convert the points to GeoViews points
    points = gv.Points([tgfs.get_coords()])

    # Create the tgf map
    tgf_map = (gv.tile_sources.OSM * points).opts(gv.opts.Points(
                global_extent=True, 
                width=1300, 
                height=900, 
                size=7,
                color='Blue',
                marker='+'))
    
    # if there are lightnings create the lighning map and add it to the TGF map
    if lightnings is not None:
        points_lightning = gv.Points([lightnings.get_coords()])
        
        lightning_map = (gv.tile_sources.OSM * points_lightning).opts(gv.opts.Points(
                global_extent=True, 
                width=1300, 
                height=900, 
                size=7,
                color='red',
                marker='+'))

        tgf_map *= lightning_map

    return tgf_map


# Print the closest lightnings
def get_nearby_lightning(tgf,lightnings:array,threshold:float=1):
    # If we are given an array of TGFs
    if type(tgf) == array:
        # Create a list to output the lighning arrays for each event
        lightnings = []

        # For all the events
        for T in tqdm(tgf):
            # Calculate the closest ones
            lightnings.append(get_nearby_lightning(T))

        return lightnings
    
    # If we are given a lightning
    elif type(tgf) == event:
        # The threshold is the maximum time to look for lightnings from the tgf
        threshold = TimeDelta(threshold,format='sec')

        # Get the TGF's timestamp
        tgf_time = tgf.timestamp

        # Get all the timestamps
        timestamps = lightnings.get_timestamps()

        # find the indices where the timedifference is less than threshold
        idx = [i for i,time in enumerate(timestamps) if abs(time - tgf_time) < threshold]

        # Get the appropriate subarray
        return lightnings[idx]

    # if it is not of type event of array then raise an error
    else:
        raise Exception("Type %s is not of type event, or array. Please use an object of type event or array for the tgf"%type(tgf))

# Give it two astropy Time objects and get back a raadpy list for the lighnings
def download_lightnings_range(start_Time:Time, end_Time:Time,VERBOSE=True):
    # Get the strings for the timestamps
    start_time  = get_epoch_time(start_Time)
    start_date  = get_epoch_date(start_Time)

    end_time    = get_epoch_time(end_Time)
    end_date    = get_epoch_date(end_Time)

    
    # Here are our login info
    payload = {
        "login_username" : "nyuad_ls",
        "login_password" : "RAADsat3U",
        "login_try" : "1"
    }

    # This will keep our session alive while we log in
    session = requests.Session()

    # Have our session logged in
    url_login = 'https://www.blitzortung.org/en/login.php'
    url = '/en/login.php'
    # result = session.get(url_login)
    # tree = html.fromstring(result.text)f
    result = session.post(
        url_login,
        data = payload
    )


    # Request the archived data
    url_archive = "https://www.blitzortung.org/en/archive_data.php?stations_users=0&selected_numbers=*&end_date="+end_date+"&end_time="+end_time+"&start_date="+start_date+"&start_time="+start_time+"&rawdata_image=0&north=90&west=-180&east=180&south=-90&map=0&width_orig=640&width_result=640&agespan=60&frames=12&delay=100&last_delay=1000&show_result=1"
    
    # Get the data website
    result = session.get(url_archive)
    tree = html.fromstring(result.content)

    # Find the iframe url
    src = 'https://www.blitzortung.org/' + np.array(tree.xpath("/html/body//iframe/@src"))[0]

    # request that url
    result = session.get(src)
    tree = html.fromstring(result.content)

    # Grab the file url:
    a = np.array(tree.xpath("/html/body//a/@href"))
    file_url = 'https://www.blitzortung.org/' + a[['archive' in url and 'raw.txt' in url for url in a]][0]

    if VERBOSE: print(bcolors.OKCYAN+'Found Lightning data at: '+bcolors.ENDC+url_archive)

    # Get the raw file and parse it
    raw  = decompress(requests.get(file_url).content).decode('utf-8').split('\n')

    if VERBOSE: print(bcolors.OKCYAN+'Data Downloaded Successfully'+bcolors.ENDC)
    
    # Create the array
    lights  = []
    # For all the lightnings in the loaded dataset
    for data in raw[1:-1]:
        # Create an event and append it to the array
        datum = data.split(',')
        lights.append(event(timestamp   = float(datum[0]) * 1e-9,
                            longitude   = in_range(float(datum[2])), 
                            latitude    = float(datum[1]),
                            detector_id = 'Blitz',
                            event_id    = datum[2],
                            mission     = 'Blitzurtong',
                            time_format = 'unix',
                            event_type  = 'Lightning'))
 
    # Return the numpy array for the file
    return array(lights)


# Give a timestamp and a threshold, and then the code will download close (in time) lightnings
def download_lightnings(event_time:Time,threshold:float = 6*60,VERBOSE=True):
    # Check if the threhsold is within the range
    if threshold <= 5*60:
        print(bcolors.WARNING+"Warning!"+bcolors.ENDC+" Threshold: %f s, is too small to be detected by Blitzortung! Using threshold = 6 * 60 s instead."%(threshold))
        threshold = 6*60

    # Get the timedelta object that corresponds to the threshold
    threshold = TimeDelta(threshold,format='sec')

    if VERBOSE:
        print(bcolors.OKCYAN+'Searching for Lightnings between:'+bcolors.ENDC+'\n\t start-time: %s\n\t end-time:   %s'
                %((event_time-threshold).to_value('iso'),(event_time+threshold).to_value('iso')))

    return download_lightnings_range(event_time-threshold,event_time+threshold,VERBOSE=VERBOSE)

# We create a function that given a bytestring extracts the ith bit:
def get_bit(i:int,string):
    '''
    Gets the ith bit from a python bytestring from the left

    Input:
    i: int --> index (frist bit is 0)
    string --> the bytestring 
    '''

    # Which byte does the bit lie into?
    byte_idx    = i//BYTE               # Integer division
    assert(byte_idx < len(string))      # Assert that the index is in the bytestring
    byte        = string[byte_idx]      # Get the appropriate byte
    bit_idx     = i - byte_idx * BYTE   # Get the index within the byte

    # Get the ith bit
    return (byte & (1 << (BYTE - bit_idx - 1))) >> (BYTE - bit_idx - 1)

# Helper function to give the index of the nth bit in a Bytestring
def get_bit_idx(n:int):
    return BYTE - 1 - n%BYTE + (n//BYTE) * BYTE

# Get range of bits
def get_bits(start:int,length:int,string):
    '''
    Gets length bits after and including index start

    Input:
    start:  int --> Start index included
    length: int --> Length of bits to obtain
    string      --> The bytestring
    '''

    # Collect the bytes and add them up
    digit_sum = 0
    for i in range(start,start+length):
        digit_sum += 2**(i-start) * get_bit(get_bit_idx(i),string)

    return digit_sum

# Create a dictionary of orbits from a file
def get_dict(filename:str,struct=ORBIT_STRUCT,condition:str=None,MAX=None):
    # Read the raw data
    file = open(filename,'rb')  # Open the file in read binary mode
    raw = file.read()           # Read all the file
    file.close()                # Close the file

    # Initialize the dictionary
    data = dict(zip(struct.keys(),[np.array(list()) for _ in range(len(ORBIT_STRUCT.keys()))]))

    # Number of bytes per line
    bytes_per_line  = sum(list(struct.values()))//8
    length          = len(raw)//bytes_per_line
    if MAX is None: MAX = length

    for i in tqdm(range(MAX),desc='Line: '):
        # Get the required number of bytes to an event
        event = raw[i*bytes_per_line:(i+1)*bytes_per_line]

        # Keep track of the number of bits read
        bits_read = 0

        # If not create an orbit
        for name,length in struct.items():
            data[name] = np.append(data[name],[get_bits(bits_read,length,event)])
            bits_read += length
    
    if condition is not None:
        try:
            idx     = np.where(eval(condition))[0]
            data    = dict(zip(struct.keys(),[arr[idx] for arr in data.values()]))
        except:
            print(bcolors.WARNING+'WARNING!' + bcolors.ENDC +' Condition ' + condition + ' is not valid for the dataset you requested. The data returned will not be filtered')

    # Return the dictionary
    return data

# Plot the dictionary data obtained from a buffer:
def plot_buffer(data,title='Plots of Buffer data'):
    # Get the keys and event numbers
    keys    = list(data.keys())
    events  = range(len(data[keys[0]]))
    colors  = cm.get_cmap('Dark2').colors

    # Create a figure
    fig, axes = plt.subplots((len(keys)+1)//2,2,sharex=True,figsize=(20,4*len(keys)//2),dpi=200)
    fig.subplots_adjust(top=0.95)
    fig.suptitle(title,fontsize=18)
    axes = axes.flatten()

    # Plot each of the data points
    for i,key,ax in zip(range(len(axes)),keys,axes):
        ax.plot(events,data[key],c=colors[i%len(colors)],lw=0.7)
        ax.set_title(key.title())

        # Customize the plot style
        ax.tick_params(axis='both',which='both',direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(axis='both', which='major', lw=0.25)
        ax.grid(axis='both', which='minor', lw=0.2, ls=':')
        

    return fig,axes

# Split the dataset in channels
def split_channels(data,struct=NONVETO_STRUCT):
    # Split the data based on their channels
    channels    = []
    for channel in np.unique(data['channel']):
        idx         = np.where(data['channel'] == channel)[0]
        channels.append(dict(zip(struct.keys(),[arr[idx] for arr in data.values()])))
    
    return channels

# Plot histograms of the energies
def plot_hists(data,struct=NONVETO_STRUCT,bins=600,RANGE=None):
    # Get the splitted channels
    channels = split_channels(data,struct)

    # Create a figure
    fig,ax  = plt.subplots(len(channels),1,figsize=(15,4*len(channels)),dpi=200,sharex=True)
    ax      = ax.flatten()
    colors  = cm.get_cmap('Dark2').colors

    # Plot the histogram of each channel
    for i,channel in enumerate(channels):
        ax[i].hist(channel['adc_counts'],bins=bins,range=RANGE,color=colors[i%len(channels)])

        ax[i].set_title('Energy of Channel: %d'%i)
        ax[i].set_yscale('log')
        ax[i].tick_params(axis='both',which='both',direction='in',top=True,right=True)
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].grid(axis='both', which='major', lw=0.25)
        ax[i].grid(axis='both', which='minor', lw=0.2, ls=':')

    return fig,ax

# Plots the timestamps of the measurements by channel
def plot_timestamps(data,struct=NONVETO_STRUCT,RANGE=None):
    # Get the splitted channels
    channels = split_channels(data,struct)

    # Create a figure
    fig,ax  = plt.subplots(len(channels),1,figsize=(15,4*len(channels)),dpi=200,sharex=True)
    ax      = ax.flatten()
    colors  = cm.get_cmap('Dark2').colors

    # Plot the histogram of each channel
    for i,channel in enumerate(channels):
        length   = len(channel['stimestamp'])
        if RANGE is None: _RANGE = (0,length)
        else: _RANGE = RANGE

        ax[i].plot   (range(*_RANGE),channel['stimestamp'][_RANGE[0]:_RANGE[1]],c=to_hex(colors[i%len(channels)]),lw=0.4)
        ax[i].scatter(range(*_RANGE),channel['stimestamp'][_RANGE[0]:_RANGE[1]],c=to_hex(colors[i%len(channels)]),marker='o',s=2)

        ax[i].set_title('Timestamp of Channel: %d'%i)
        ax[i].tick_params(axis='both',which='both',direction='in',top=True,right=True)
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
        ax[i].grid(axis='both', which='major', lw=0.25)
        ax[i].grid(axis='both', which='minor', lw=0.2, ls=':')

    return fig,ax
