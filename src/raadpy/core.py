#############################
#     RAAD CORE Library     #
#############################

# Import necessary Libraries
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import to_hex
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime as dt
from lxml import html
from gzip import decompress
from tqdm.notebook import tqdm
import json
import requests
from requests.auth import HTTPBasicAuth
from requests.auth import HTTPDigestAuth
from requests_oauthlib import OAuth1
import os
import csv
from IPython.display import clear_output as clear
import pymysql
from sshtunnel import SSHTunnelForwarder
import paramiko
CUPY_AVAILABLE = False
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ModuleNotFoundError:
    print('\033[93mWarning!\033[95m Cupy not found! GPU accelleration is not available\033[0m\n')

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

PWR_ON_CMD              = 'txrx 4 14 600 0007010100'
PWR_OFF_CMD             = 'txrx 4 14 600 0007000000'

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

TGF_STRUCT={
    'stimestamp': 48,
    'channel': 2,
    'adc_counts': 14,
}

ORBIT_UNITS     = ['(s)','(C)','(Hz)','(Hz)','(Hz)','(Hz)','(Hz)','(DAC ch)','(DAC ch)','(PMT 0 - SiPM 1)','(OFF 0 - ON 1)',' ',' ',' ']
VETO_UNITS      = ['','','','(ms)']
NONVETO_UNITS   = ['','','(ms)']

SIPM            = 12
PMT             = 13
BOTH            = 4

HOST            = "https://light1.mcs.nanoavionics.com"
TOKEN           = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoia2hhbGlmYSIsImV4cCI6MTcwNDA2NzIwMCwiZW1haWwiOiJhZGcxMUBueXUuZWR1In0.LiV8bfKb2JUG2eIIxouXKebQpPFLXewO1BqoOD22xS4"

# Deciphering the command list
CMND_LIST = {
"4"  : {"14":{  "description" : "Emergency Power-off of PAYLOAD if OBC reboots",
                "number_in_graph" : 0,
                "cmd_number" : 33}
                        },

"12" : { 
        "4": {  "08":    {"description" : "Suspend Payload as well PMT",
                        "number_in_graph" : 12,
                        "cmd_number" : 36},

                "01":    {"description" : "Load Scenario Suspended, PMT PAYLOAD",
                        "number_in_graph" : 12,
                        "cmd_number" : 25},

                "02":    {"description" : "Load Scenario Awaken, PMT PAYLOAD",
                        "number_in_graph" : 11,
                        "cmd_number" : 27},

                "04":    {"description" : "Reboot PMT Payload",
                        "number_in_graph" : 13,
                        "cmd_number" : 29}
                },

        "8": {  "0F":   {"description" : "After an Emergency Shutdown happened to PMT",
                        "number_in_graph" : 3,
                        "cmd_number" : 34},

                "00":   {"description" : "Load Scenario 0 (Default) PMT PAYLOAD",
                        "number_in_graph" : 0,
                        "cmd_number" : 1},

                "01":   {"description" : "Load Scenario 1 PMT PAYLOAD",
                        "number_in_graph" : 1,
                        "cmd_number" : 2},

                "02":   {"description" : "Load Scenario 2 PMT PAYLOAD",
                        "number_in_graph" : 2,
                        "cmd_number" : 3},

                "03":   {"description" : "Load Scenario 3 PMT PAYLOAD",
                        "number_in_graph" : 3,
                        "cmd_number" : 4},

                "04":   {"description" : "Load Scenario 4 PMT PAYLOAD",
                        "number_in_graph" : 4,
                        "cmd_number" : 5},

                "05":   {"description" : "Load Scenario 5 PMT PAYLOAD",
                        "number_in_graph" : 5,
                        "cmd_number" : 6},

                "06":   {"description" : "Load Scenario 6 PMT PAYLOAD",
                        "number_in_graph" : 6,   
                        "cmd_number" : 7},

                "07":   {"description" : "Load Scenario 7 PMT PAYLOAD",
                        "number_in_graph" : 7,
                        "cmd_number" : 8},

                "08":   {"description" : "Load Scenario 8 PMT PAYLOAD",
                        "number_in_graph" : 8,
                        "cmd_number" : 9},

                "09":   {"description" : "Load Scenario 9 PMT PAYLOAD",
                        "number_in_graph" : 9,
                        "cmd_number" : 10},

                "10":   {"description" : "Load Scenario 10 PMT PAYLOAD",
                        "number_in_graph" : 10,
                        "cmd_number" : 11}
                },

        "9": {  "0F":   {"description" : "Load CUSTOM Scenario PMT PAYLOAD",
                        "number_in_graph" : 14,
                        "cmd_number" : 23}},


        "1": {  "00":   {"description" : "Ping PMT PAYLOAD",
                        "number_in_graph" : 15,
                        "cmd_number" : 31}}
        },

"13" : { 
        "4": {  "08":    {"description" : "Suspend Payload as well SiPM",
                        "number_in_graph" : 12,
                        "cmd_number" : 37},

                "01":    {"description" : "Load Scenario Suspended, SiPM PAYLOAD",
                        "number_in_graph" : 12,
                        "cmd_number" : 26},

                "02":    {"description" : "Load Scenario Awaken, SiPM PAYLOAD",
                        "number_in_graph" : 11,
                        "cmd_number" : 28},

                "04":    {"description" : "Reboot SiPM Payload",
                        "number_in_graph" : 13,
                        "cmd_number" : 30},
                },

        "8": {  "0F":   {"description" : "After an Emergency Shutdown happened to SiPM",
                        "number_in_graph" : 3,
                        "cmd_number" : 35},

                "00":   {"description" : "Load Scenario 0 (Default) SiPM PAYLOAD",
                        "number_in_graph" : 0,
                        "cmd_number" : 12},

                "01":   {"description" : "Load Scenario 1 SiPM PAYLOAD",
                        "number_in_graph" : 1,
                        "cmd_number" : 13},

                "02":   {"description" : "Load Scenario 2 SiPM PAYLOAD",
                        "number_in_graph" : 2,
                        "cmd_number" : 14},

                "03":   {"description" : "Load Scenario 3 SiPM PAYLOAD",
                        "number_in_graph" : 3,
                        "cmd_number" : 15},

                "04":   {"description" : "Load Scenario 4 SiPM PAYLOAD",
                        "number_in_graph" : 4,
                        "cmd_number" : 16},

                "05":   {"description" : "Load Scenario 5 SiPM PAYLOAD",
                        "number_in_graph" : 5,
                        "cmd_number" : 17},

                "06":   {"description" : "Load Scenario 6 SiPM PAYLOAD",
                        "number_in_graph" : 6,
                        "cmd_number" : 18},

                "07":   {"description" : "Load Scenario 7 SiPM PAYLOAD",
                        "number_in_graph" : 7,
                        "cmd_number" : 19},

                "08":   {"description" : "Load Scenario 8 SiPM PAYLOAD",
                        "number_in_graph" : 8,
                        "cmd_number" : 20},

                "09":   {"description" : "Load Scenario 9 SiPM PAYLOAD",
                        "number_in_graph" : 9,
                        "cmd_number" : 21},

                "10":   {"description" : "Load Scenario 10 SiPM PAYLOAD",
                        "number_in_graph" : 10,
                        "cmd_number" : 22}
                },

        "9": {  "0F":   {"description" : "Load CUSTOM Scenario SiPM PAYLOAD",
                        "number_in_graph" : 14,
                        "cmd_number" : 24}},


        "1": {  "00":   {"description" : "Ping SiPM PAYLOAD",
                        "number_in_graph" : 15,
                        "cmd_number" : 32}}

        }}

##################################################################################
# Helper functions
##################################################################################

# helper function to convert longitude in range -180 to 180
def in_range(longitude):
    return longitude if longitude <= 180 else longitude - 360

# Get a list and return its unique elements
def unique(l:list):
    return list(dict.fromkeys(l))

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

# Get the most significant bit of number
def get_msb(x:int,BITS=48):
    for i in range(BITS):
        x |= x >> int(2**i)
    return (x+1) >> 1

# Get the subdict from an array dict
def subdict(dict:dict,min:int=None,max:int=None):
    new_dict = {}
    for key in dict: new_dict[key] = dict[key][min:max]
    
    return new_dict

# Get the two arrays of different size and match their elements
def match(*arrays,from_end=True):
    # Length of the final arrays
    length = min([len(array) for array in arrays])

    # Concatenate the arrays
    if from_end:
        return [array[-length:] for array in arrays]
    else: 
        return [array[:length] for array in arrays]

# Collect timestamps of command execution
def collect_time_cmd(log:dict,cmd,include_end:bool=True):
    return np.array([line['timestamp'] for line in log if cmd in line['command']])

# Go through the logfile and check if the starts and ends match
def get_unmatched_cycles(log:list):
    EXPECT = None
    UNMATCHED = []
    for i in range(len(log)-1,-1,-1):
        # If this is a power command
        if PWR_OFF_CMD in log[i]['command'] or PWR_ON_CMD in log[i]['command']:
            if EXPECT is None: EXPECT = PWR_OFF_CMD if PWR_OFF_CMD in log[i]['command'] else PWR_ON_CMD

            # If you see something out of cycle
            if EXPECT not in log[i]['command']:
                # Add it to the mismatched commands
                UNMATCHED.append(log[i])
                EXPECT = PWR_OFF_CMD if EXPECT == PWR_ON_CMD else PWR_ON_CMD

            # Change what you expect
            EXPECT = PWR_OFF_CMD if EXPECT == PWR_ON_CMD else PWR_ON_CMD
    
    return UNMATCHED

# Get the subfiles and put the in a dictionary
def get_filenames(raw_dir:str = './'):
    fnames      = os.listdir(raw_dir)
    filenames   = {}
    for i in range(1,10): 
        res = [name for name in fnames if f'buff{i}' in name]
        if len(res)>0: filenames[f'buff{i}'] = res[0]
    for name in fnames: 
        if 'log.txt' in name: filenames['log'] = name

    return filenames

# Plot Logfile timestamps for testing (particularly the analysis workshop)
def plot_timestamps_log(start_timestamps_log,end_timestamps_log,UNMATCHED = [],fig=None,ax=None,figsize=(9,5),**kwargs):
    if fig is None: fig = plt.figure(figsize=figsize,**kwargs)
    if ax  is None: ax  = fig.add_subplot(111)

    # Plot the stuff
    ax.plot(start_timestamps_log,c='g',lw=0.5,label='POWER ON TIME')
    ax.plot(end_timestamps_log,  c='r',lw=0.5, label='POWER OFF TIME')
    ax.scatter(list(range(len(start_timestamps_log))),start_timestamps_log,c='g',lw=0.5,s=0.5)
    ax.scatter(list(range(len(end_timestamps_log))),end_timestamps_log,  c='r',lw=0.5,s=0.5)
    for line in UNMATCHED:
        ax.axhline(line['timestamp'],ls=':',c='lightgreen',lw=0.5)

    # Prettify
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Timestamp")
    ax.legend(frameon=False,loc='upper left')

    return fig, ax

# Analysis workshop function
def get_log_by_number(number:int):
    '''Returns the ith log file and saves it to the current working directory and another .backup directory'''
    
    # Check if the file already exists
    if os.path.exists(os.path.join('.',f'RAAD-{number}-log.txt')): 
        print(bcolors.WARNING+"You have already downloaded this file! Delete it and run again if you want to replace it!")
        return

    # Download the logfile and save
    request = requests.get(f'http://arneodolab.abudhabi.nyu.edu:8000/logfix/split/RAAD-{number}-log.txt')
    with open(f'RAAD-{number}-log.txt','w') as file: file.write(request.content.decode('utf-8'))

def log_submit(number:int):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect('arneodolab.abudhabi.nyu.edu', username='raad',password='nyuad123$')

    #Setup sftp connection and transmit this script 
    sftp = client.open_sftp() 
    sftp.put(f'./RAAD-{number}-log.txt', f'/home/raad/logfix/results/RAAD-{number}-log.txt')

    sftp.close()

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

# Send a data download request using REST
class RestOperations:
    """Send a data download request using the REST Protocol
    """
    # Initialize with the link
    def __init__(self, apiEndPoint, **kwargs):
        """Constructor

        Args:
            apiEndPoint (string): the url needed to make the request
        """
        self.apiEndPoint = apiEndPoint
        self.kwargs = kwargs
    
    def SendGetReq(self):
        """Send a download request to the URL

        Returns:
            json: A json file with all the downloaded data
        """
        # Get the needed authorization information
        auth = self.CallAuth(self.kwargs)

        # Make the request
        RespGetReq = requests.get(self.apiEndPoint, auth = auth, stream=True)

        # Check for errors
        if RespGetReq.status_code != 200:
            RespGetReq.raise_for_status()
            raise RuntimeError(f"Request to {self.apiEndPoint} returned status code {RespGetReq.status_code}")

        # Convert the output to a json and return
        return json.loads(RespGetReq.text)

    def CallAuth(self, OptionalAttrs):
        """Handle authorization stuff

        Args:
            OptionalAttrs (_type_): The necessary arguments needed for the type of authorization

        Returns:
            auth: An authorization object
        """
        authType = self.ValidateAuthAttrs(OptionalAttrs)
        if not authType:
            auth = None            
        elif authType == 'token':
            auth = HTTPBearerAuth(OptionalAttrs.get('token'))
        elif authType == 'basic':
            auth = HTTPBasicAuth(OptionalAttrs.get('username'), OptionalAttrs.get('password'))
        elif authType  == 'digest':
            auth = HTTPDigestAuth(OptionalAttrs.get('username'), OptionalAttrs.get('password'))
        elif authType  == 'oa1':
            auth = OAuth1(OptionalAttrs.get('AppKey'), OptionalAttrs.get('AppSecret'), OptionalAttrs.get('UserToken'), OptionalAttrs.get('UserSecret'))
        return auth
    
    def ValidateAuthAttrs(self, OptionalAttrs):
        """Make sure the optinal attributes of this class exist
        """
        if 'authType' not in OptionalAttrs:
            authType = None
        else:
            if OptionalAttrs.get('authType') not in ['token', 'digest', 'basic', 'oa1']:
                raise ValueError("Unknown authType received", OptionalAttrs.get('authType'))
            else:
                if OptionalAttrs.get('authType') == 'token' and 'token' not in OptionalAttrs:
                    raise ValueError("authType 'token' requires token")
                elif OptionalAttrs.get('authType') == 'basic' and not all(attr in OptionalAttrs for attr in ['username', 'password']):
                    raise ValueError("authType 'basic' requires username, password")
                elif OptionalAttrs.get('authType') == 'digest' and not all(attr in OptionalAttrs for attr in ['username', 'password']):
                    raise ValueError("authType 'digest' requires username, password")
                elif OptionalAttrs.get('authType') == 'oa1' and not all(attr in OptionalAttrs for attr in ['AppKey', 'AppSecret', 'UserToken' 'UserSecret']):
                    raise ValueError("authType 'oa1' requires AppKey, AppSecret, UserToken, UserSecret")
                else:
                    authType = OptionalAttrs.get('authType')
        return authType

class HTTPBearerAuth(requests.auth.AuthBase):
    '''requests() does not support HTTP Bearer tokens authentication, create one'''
    def __init__(self, token):
        self.token = token
    def __eq__(self, other):
        return self.token == getattr(other, 'token', None)
    def __ne__(self, other):
        return not self == other
    def __call__(self, r):
        r.headers['Authorization'] = 'Bearer ' + self.token
        return r