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

ORBIT_UNITS     = ['(s)','(C)','(Hz)','(Hz)','(Hz)','(Hz)','(Hz)','(DAC ch)','(DAC ch)','(PMT 0 - SiPM 1)','(OFF 0 - ON 1)',' ',' ',' ']
VETO_UNITS      = ['','','','(ms)']
NONVETO_UNITS   = ['','','(ms)']

HOST            = "https://light1.mcs.nanoavionics.com"
TOKEN           = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoia2hhbGlmYSIsImV4cCI6MTcwNDA2NzIwMCwiZW1haWwiOiJhZGcxMUBueXUuZWR1In0.LiV8bfKb2JUG2eIIxouXKebQpPFLXewO1BqoOD22xS4"

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