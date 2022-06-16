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
import requests
from lxml import html
from gzip import decompress
from tqdm.notebook import tqdm

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