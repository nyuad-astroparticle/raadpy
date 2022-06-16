#############################
#     RAAD Functionality    #
#############################

from .core import *
from .__array import array
from .event import *

# Print the closest lightnings
def get_nearby_lightning(tgf,lightnings:array,threshold:float=1):
    # If we are given an array of TGFs
    if type(tgf) == array:
        # Create a list to output the lighning arrays for each event
        lights = []

        # For all the events
        for T in tqdm(tgf,desc='Event'):
            # Calculate the closest ones
            lights.append(get_nearby_lightning(T,lightnings,threshold))

        lights = [light for sublist in lights for light in sublist]
        return array(unique(lights))
    
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
        return array(lightnings[idx])

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
def get_bits(start:int,length:int,string,STUPID:bool=False):
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
        bit = get_bit(get_bit_idx(i),string) if not STUPID else get_bit(2*start+length -i-1,string)
        digit_sum += 2**(i-start) * bit

    return digit_sum

# Create a dictionary of orbits from a file
def get_dict(filename:str,struct=ORBIT_STRUCT,condition:str=None,MAX=None,STUPID:bool=False):
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
            data[name] = np.append(data[name],[get_bits(bits_read,length,event,STUPID=STUPID)])
            bits_read += length
    
    if condition is not None:
        try:
            idx     = np.where(eval(condition))[0]
            data    = dict(zip(struct.keys(),[arr[idx] for arr in data.values()]))
        except:
            print(bcolors.WARNING+'WARNING!' + bcolors.ENDC +' Condition ' + condition + ' is not valid for the dataset you requested. The data returned will not be filtered')

    # Return the dictionary
    return data

# Corrects the timestamp based on orbit rate
def correct_time_orbit(orbit:dict,TIME:int=20,RANGE=(0,100)):
    # Some variables
    total_cnt   = 0     # Stores the total number of events
    timestamp   = [0]   # New timestamp

    # For each count in the orbit
    for count in orbit['ratev'][RANGE[0]:RANGE[1]]:
        # Get the next number of counts
        count *= TIME
        if count == 0:
            timestamp[-1] += TIME
            continue

        # Linearly distribute the timestamps in between
        for item in np.linspace(timestamp[-1],timestamp[-1] + TIME,int(count)+1)[1:]: timestamp.append(item)
        total_cnt += count

    # remove the last element of the timestamp
    timestamp = timestamp[:-1]

    # Fix the total number of entries we have
    total_cnt = int(total_cnt)

    return timestamp, total_cnt

# To auditionally correct for the rest of the data we want to so using the stimestamp
# Correct based on FPGA counter
def correct_time_FPGA(data:dict,RIZE_TIME:int=1,CONST_TIME:int=1,TMAX:int=10000-1,RANGE=(0,1600),return_endpoints:bool=False):
    # Find all the ramps
    # Array to store the beginning each ramp
    starting = []

    # Find all the starting points
    for i in range(RANGE[0],RANGE[1]-2):
        # Get the triplet
        A = data['stimestamp'][i]
        B = data['stimestamp'][i+1]
        
        # Examine cases
        if B-A < 0: starting.append(i+1)

    # Array to store the endings of each ramp
    ending = []

    # Find all the ending points
    for i in range(RANGE[0],RANGE[1]-2):
        # Get the triplet
        A = data['stimestamp'][i]
        B = data['stimestamp'][i+1]
        C = data['stimestamp'][i+2]

        # Examine cases
        if C-B < 0 and B-A != 0: 
            if B==TMAX: ending.append(i)
            else: ending.append(i+1)
        
        elif A == B and B != TMAX and C-B < 0: ending.append(i+1)

        elif C==B and B==TMAX and B-A > 0: ending.append(i)

    # Add the first point
    if starting[0] > ending[0]: starting.insert(0,RANGE[0])

    # Create the pairs of start and end points
    ramps = list(zip(starting,ending))

    # Now that we have all the ramps we assign one second to each ramp and we place the points accordingly
    curr_second = 0     # Current second
    timestamp   = []    # Timestamps
    valid_data  = []    # List to store the data on the rize or fall

    # For each ramp
    for ramp in ramps:
        # Take the elements of the ramp and append them to timestamp
        for i in range(ramp[0],ramp[1]+1):
            timestamp.append(curr_second+data['stimestamp'][i]*RIZE_TIME/(TMAX+1))
            valid_data.append(i)

        # Increase the timestamp
        curr_second+=RIZE_TIME+CONST_TIME
    
    if return_endpoints: return timestamp, valid_data, ramps
    return timestamp, valid_data 

# Now putting everything together
def correct_time(data:dict,orbit:dict,TIME:int=20,RANGE_ORBIT=(0,100),RIZE_TIME:int=1,CONST_TIME:int=1,TMAX:int=10000-1):
    # First collect the timstamp based on the orbit data
    # Some variables
    total_cnt       = 0                     # Stores the total number of events
    processed_cnt   = 0                     # Stores the number of events processed
    current_time    = TIME*RANGE_ORBIT[0]   # The current time 
    timestamp       = []                    # New timestamp
    valid_events    = []                    # Stores the indices of the events that can be timestamped

    oops = 0
    # For each count in the orbit
    for count in orbit['ratev'][RANGE_ORBIT[0]:RANGE_ORBIT[1]]:
        # Get the next number of counts
        count = int(count*TIME)
        if count == 0:
            current_time += TIME
            continue

        # Now filter the events that can be placed in the timestamp and
        timestamp_veto, valid_data = correct_time_FPGA(data,RIZE_TIME=RIZE_TIME,CONST_TIME=CONST_TIME,TMAX=TMAX,RANGE=(processed_cnt,processed_cnt+count))

        # Add the new data on the timestamp
        for valid,time in zip(valid_data,timestamp_veto):
            timestamp.append(current_time + time)
            valid_events.append(valid)
            
        # Update the current time to the last used time
        if timestamp[-1] - current_time > TIME: 
            # print('Oops: ',oops,current_time,timestamp[-1])
            oops+=1
            current_time = timestamp[-1]
            # current_time += TIME
        else:
            current_time += TIME
        
        # Update the total count
        total_cnt       += len(valid_data)
        processed_cnt   += count

    print("Oops': ",oops/(RANGE_ORBIT[1]-RANGE_ORBIT[0]))

    # # remove the last element of the timestamp
    # timestamp = timestamp[:-1]

    # Fix the total number of entries we have
    total_cnt = int(total_cnt)

    return timestamp, total_cnt, valid_events
