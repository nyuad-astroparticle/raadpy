#############################
#     RAAD Functionality    #
#############################

from time import daylight
from .core import *
from .rparray import array
from .event import *

# Print the closest lightnings
def get_nearby_lightning(tgf,lightnings:array,threshold:float=1):
    """Given an array, or a single event object filter a raadpy array that contains lightnings within a threshold.

    Args:
        tgf (_type_): A single Event object, or an array of events of which to find the near lightnings
        lightnings (array): Raadpy array of lightnings of which to filter
        threshold (float, optional): Threshold in time to filter the lightings. Defaults to 1.

    Returns:
        lightnings (array): A filtered array of lightnings
    """

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
    """Download lightnings in a given time range from blitzortung.com

    Args:
        start_Time (Time): The starting time of the events
        end_Time (Time): The ending time of the events
        VERBOSE (bool, optional): Print description if needed. Defaults to True.

    Returns:
        lightnings (array): Ligtnings in time range
    """
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
    """Given an event time download lightings around it for a given time threshold

    Args:
        event_time (Time): Timestamp of the event
        threshold (float, optional): Seconds around the event time to look for lightnings. Defaults to 6*60.
        VERBOSE (bool, optional): Print a description of the process. Defaults to True.

    Returns:
        lightnings (array): The array of lightnings downloaded.
    """
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
def get_dict(filename:str,struct=ORBIT_STRUCT,condition:str=None,MAX=None,STUPID:bool=False,VERIFY=False,threshold=5e-5):
    """Decode the data of a buffer with a given structure into a dictionary

    Args:
        filename (str): The filename where the buffer is
        struct (_type_, optional): The structure of the bits of the buffer represented in a dictionary. Defaults to ORBIT_STRUCT.
        condition (str, optional): If you want you can add a condition such as data['id_bit']==1 to filter the data as they're being loaded. Defaults to None.
        MAX (_type_, optional): Maximum number of lines to read, if None then read all of them. Defaults to None.
        STUPID (bool, optional): Should be set to True if you are reading VETO and NONVETO. Defaults to False.

    Returns:
        data (dict): Dictionary with the decoded arrays of measurements
    """
    # Read the raw data
    file = open(filename,'rb')  # Open the file in read binary mode
    raw = file.read()           # Read all the file
    file.close()                # Close the file

    # Initialize the dictionary
    data = dict(zip(struct.keys(),[np.array(list()) for _ in range(len(ORBIT_STRUCT.keys()))]))

    # Number of bytes per line
    bytes_per_line  = sum(list(struct.values()))//8
    length          = len(raw)//bytes_per_line
    if MAX is None or MAX > length: MAX = length

    # Check if VERIFICATION can occur
    if VERIFY:
        # If you can't correct then don't
        if 'stimestamp' not in struct.keys():
            VERIFY = False
            
        # Define the threshold where a tiemstamp difference is just too much
        THRESHOLD = 0
        for i in range(struct['stimestamp']+1):THRESHOLD += 2**i
        if threshold <= 1:
            THRESHOLD *= threshold
        else:
            THRESHOLD = threshold

        # print("%.4e"%THRESHOLD)

    # Current byte index in the file
    curr = 0
    with tqdm(total=MAX,desc='Line: ') as pbar:
        # Index of line
        i = 0
        while i < MAX:
            update = 1
            # Get the required number of bytes to an event
            # event = raw[i*bytes_per_line:(i+1)*bytes_per_line]
            event = raw[curr:curr + bytes_per_line]

            # if you reached the end of the file break
            if len(event) < bytes_per_line: 
                pbar.update(MAX-i)
                break

            # Keep track of the number of bits read
            bits_read = 0

            # If not create an orbit
            for name,length in struct.items():
                data[name] = np.append(data[name],[get_bits(bits_read,length,event,STUPID=STUPID)])
                bits_read += length

            # Verify the datum makes sense
            if VERIFY:
                # If there are more than two datapoints in the timestamp
                if len(data['stimestamp'])>=2:
                    # If the difference between the last two timestmaps is absurd
                    if abs(data['stimestamp'][-1] - data['stimestamp'][-2]) > THRESHOLD:# or \
                       #(data['stimestamp'][-1] - data['stimestamp'][-2] < 0) and (abs(data['stimestamp'][-2] - data['stimestamp'][-1]) < THRESHOLD/20):
                        # remove the previous datapoint
                        for key in data.keys():
                            data[key] = np.delete(data[key],-1)

                        # Move forward by two bytes
                        curr   -= bytes_per_line - 2
                        i      -= 1
                        update  = 0

            # Update reader position
            curr    += bytes_per_line
            i       += 1
            pbar.update(update)

        
    # If you want to filter, then apply the filter to the loaded data directly
    if condition is not None:
        try:
            idx     = np.where(eval(condition))[0]
            data    = dict(zip(struct.keys(),[arr[idx] for arr in data.values()]))
        except:
            print(bcolors.WARNING+'WARNING!' + bcolors.ENDC +' Condition ' + condition + ' is not valid for the dataset you requested. The data returned will not be filtered')

    # Specific loading changes
    if 'temperature' in struct.keys():
        data['temperature'] -= 55
        
    # Return the dictionary
    return data

# Corrects the timestamp based on orbit rate
def correct_time_orbit(orbit:dict,key:str='rate0',TIME:int=20,RANGE=(0,100)):
    """Corrects the time of events based on the data of the orbit buffer

    Args:
        orbit (dict): The orbit buffer
        key (str): Key for the corresponding event buffer rate
        TIME (int, optional): The period of the rate measurements. Defaults to 20.
        RANGE (tuple, optional): a range of indices to translate of the orbit buffer. Defaults to (0,100).

    Returns:
        timestamp (np.array): Array with the corrected timestamps
        start_cnt (int): Starting index on the corresponding buffer
        end_cnt (int): Ending index on the corresponding buffer
    """
    # Some variables
    start_cnt   = 0
    end_cnt     = 0     # Stores the total number of events
    timestamp   = [0]   # New timestamp

    # Start counting events from the correct timestamp
    if RANGE[0] != 0:
        for counts in orbit[key][0:RANGE[0]]:
            start_cnt += int(counts * TIME)
        
        # Start counting from this value
        end_cnt += start_cnt

    # For each count in the orbit
    for count in orbit[key][RANGE[0]:RANGE[1]]:
        # Get the next number of counts
        count = int(count*TIME)
        if count == 0:
            timestamp[-1] += TIME
            continue

        # Linearly distribute the timestamps in between
        for item in np.linspace(timestamp[-1],timestamp[-1] + TIME, int(count)+1)[1:]: timestamp.append(item)
        end_cnt += count

    # remove the last element of the timestamp
    timestamp = timestamp[:-1]

    # Fix the total number of entries we have
    end_cnt = int(end_cnt)

    return timestamp, start_cnt, end_cnt

# detect every time the slope is negative
def get_ramps(data:dict):
    # Store the indices of the negative slopes
    idx = []

    # For each point
    for i in range(len(data)-2):
        # If the slope is negative
        if data[i+1]-data[i] < 0:
            # Append the index of the point
            idx.append(i+1)
    
    # Create tuples with the start and end of each ramp
    if len(idx) > 0: ramps = [(0,idx[0])]
    else: return [(0,len(data))]
    for i in range(1,len(idx)):
        ramps.append((idx[i-1],idx[i]))

    return ramps


# Obtain a subset of data from a dictionary
def dict_subsec(data:dict,idx:list):

    # Copy the dictionary
    new_dict = {}

    # For each key of the data, append the new version
    for key in data.keys():
        new_dict[key] = data[key][idx]
    
    return new_dict

# To auditionally correct for the rest of the data we want to so using the stimestamp
# Correct based on FPGA counter
def correct_time_FPGA(data:dict,RIZE_TIME:float=1,CONST_TIME:float=1,TMAX:int=10000-1,RANGE=(0,1600),return_endpoints:bool=False):
    """Correct the time on the VETO or NONVETO buffer according to FPGA counter reconstruction

    Args:
        data (dict): The buffer data
        RIZE_TIME (int, optional): Time in seconds it takes for the FPGA to rize. Defaults to 1.
        CONST_TIME (int, optional): Time in seconds it takes for the FPGA to reset after it has risen to the saturation value. Defaults to 1.
        TMAX (int, optional): The staturation value of the FPGA. Defaults to 10000-1.
        RANGE (tuple, optional): The indices on the buffer to correct within. Defaults to (0,1600).
        return_endpoints (bool, optional): Return the start and end indices of the selected events. Defaults to False.

    Returns:
        timestamp (np.array): Array with the corrected timstamp for each valid entry
        valid_entries (list): Indices of the valid entries within the dataset (AKA. The nonsaturated entries)
        ramps (np.array): Array of tuples each with the start and end of a rising segment
    """
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
    if (len(starting)!=0 and len(ending)!=0) and starting[0] > ending[0]: starting.insert(0,RANGE[0])

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
    
    if return_endpoints: return timestamp, valid_data, np.array(ramps)
    return timestamp, valid_data

# Now putting everything together
def correct_time(data:dict,orbit:dict,key:str='rate0',TIME:int=20,RANGE_ORBIT=(0,100),RIZE_TIME:float=1,CONST_TIME:float=1,TMAX:int=10000-1):
    """Correct time using both FPGA and Orbit corrections simultaneously and generate a timestamp for the valid_data

    Args:
        data (dict): The data buffer to correct the timestamp of
        orbit (dict): The corresponding orbit buffer
        key (str): Key for the corresponding event buffer rate
        TIME (int, optional): Period of the orbit buffer measurments. Defaults to 20.
        RANGE_ORBIT (tuple, optional): The range of indices in the orbit buffer to translate. Defaults to (0,100).
        RIZE_TIME (int, optional): The time it takes for the FPGA counter to saturate. Defaults to 1.
        CONST_TIME (int, optional): The time the FPGA counter spends saturated. Defaults to 1.
        TMAX (int, optional): The maximum value of the FPGA counter. Defaults to 10000-1.

    Returns:
        timestamp (np.array): New timestamp values
        total_cnt (int): Number of datapoints translated
        valid_events (list): list of indices of valid events
    """
    # First collect the timstamp based on the orbit data
    # Some variables
    total_cnt       = 0                     # Stores the total number of events
    processed_cnt   = 0                     # Stores the number of events processed
    current_time    = TIME*RANGE_ORBIT[0]   # The current time 
    timestamp       = []                    # New timestamp
    valid_events    = []                    # Stores the indices of the events that can be timestamped

    # Start counting events from the correct timestamp
    if RANGE_ORBIT[0] != 0:
        for counts in orbit[key][0:RANGE_ORBIT[0]]:
            processed_cnt += int(counts * TIME)

    # Error flag
    oops = 0
    # For each count in the orbit
    for count in orbit[key][RANGE_ORBIT[0]:RANGE_ORBIT[1]]:
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

    if oops != 0: print("Oops': ",oops/(RANGE_ORBIT[1]-RANGE_ORBIT[0]))

    # # remove the last element of the timestamp
    # timestamp = timestamp[:-1]

    # Fix the total number of entries we have
    total_cnt = int(total_cnt)

    return timestamp, total_cnt, valid_events


# Download a range of data based on some limit
def download_range(url:str,token,limit:int=5000,VERBOSE:bool=False):
    """Downloads a range of data given a url and a token from the NA servers. 
    Automatically handles large file sizes.

    Args:
        url (str): the url from the NA server with the data to download from 
        token (str): The string value of the token for security authentication
        limit (int, optional): Number of rows to download at one go. Large numbers make the server crash. Defaults to 5000.
        VERBOSE (bool, optional): If true update statistics are printed while the fies is being downloaded. Defaults to False.

    Returns:
        data (list): a list of the binary strings of the downloaded data
    """

    # store the result
    data        = []
    last_data   = []
    seq         = -1
    cnt         = 0

    # Keep downloading until there is nothing left
    while True:
        # Print how much data you have downloaded
        clear(wait=True)
        if VERBOSE: 
            print('Current File: ',url,'\nEntries Downloaded:',len(data),'\nLast Sequence Number:',seq,'\nIterations:',cnt)
            # find the number of bytes per entry
            print('Bytes per entry: ',np.unique([len(d) for d in data]))
            cnt+=1

        # Do the REST stuff
        rest = RestOperations(url+f'&limit={limit}&seq_nr=gte.{seq}', authType = 'token', token = token)
       
        # Download the data
        last_data   = rest.SendGetReq()
        data        += last_data

        # If there are no more data exit
        if len(last_data) < limit or seq == max([datum['seq_nr'] for datum in data]):
            return data
        
        # Find the last sequence number
        seq = max([datum['seq_nr'] for datum in data])

# Order the data according to entry number
def sort(data,field='entry_nr'):
    """Sort the data based on a metadata field

    Args:
        data (array of dictionaries): The array of dictionaries from the downloaded data
        field (str, optional): The metadata field to sort according to. Defaults to 'entry_nr'.

    Returns:
        sorted: Sorted list of lists
    """
    if len(data) <= 1: return data
    
    # Get the indices
    idx = np.argsort([d[field] for d in data])
    
    # Sorted array
    sorted = [data[idx[i]] for i in range(len(data))]

    return sorted

# Download data based on various keys
def download_file_ver(buffer:int = 1, file_ver=1):
    """Download a data from NA server with a common file version

    Args:
        buffer (int, optional): The buffer to download. Defaults to 1.
        file_ver (int, optional): The file version number. Defaults to 1.

    Returns:
        data: list of dictionaries with the rows
    """
    # Generate some variables
    fileName="pc_buff"+str(buffer)
    host="https://light1.mcs.nanoavionics.com"
    token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoia2hhbGlmYSIsImV4cCI6MTcwNDA2NzIwMCwiZW1haWwiOiJhZGcxMUBueXUuZWR1In0.LiV8bfKb2JUG2eIIxouXKebQpPFLXewO1BqoOD22xS4"
    url = f'{host}/{fileName}_download?file_ver=eq.{file_ver}'

    # Download the data using segmented download
    data = download_range(url,token,VERBOSE=True)

    # Sort the data
    data = sort(data)

    return data

# Download data based on various keys
def download_log(start:str=None,end:str=None):
    """Download a log file from the NA version

    Args:
        file_ver (int, optional): The file version number. Defaults to 1.

    Returns:
        data: list of dictionaries with the rows
    """
    # Generate some variables
    fileName="pc_se0_log"
    host="https://light1.mcs.nanoavionics.com"
    token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoia2hhbGlmYSIsImV4cCI6MTcwNDA2NzIwMCwiZW1haWwiOiJhZGcxMUBueXUuZWR1In0.LiV8bfKb2JUG2eIIxouXKebQpPFLXewO1BqoOD22xS4"
    url = f'{host}/{fileName}_download?'
    if start is not None: 
        url += f'archived_ts=gte.{start}'
        if end is not None: url += f'&archived_ts=lt.{end}'
    elif end is not None: url += f'archived_ts=lt.{end}'

    # Download the data using segmented download
    data = download_range(url,token,VERBOSE=True)

    # Sort the data
    data = sort(data)

    return data

# Download data based on time range
def download_time_delta(buffer:int = 1, start:str=None, end:str=None):
    """Download NA data on a time interval 

    Args:
        buffer (int, optional): The buffer number. Defaults to 1.
        start (str, optional): String with iso date to start. Defaults to '2022-06-01T00:00:00'.
        end (str, optional): String with iso date to end. Defaults to '2022-06-07T00:00:00'.

    Returns:
        data: list of dictionaries with the rows
    """
    # Generate some variables
    fileName="pc_buff"+str(buffer)
    host="https://light1.mcs.nanoavionics.com"
    token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoia2hhbGlmYSIsImV4cCI6MTcwNDA2NzIwMCwiZW1haWwiOiJhZGcxMUBueXUuZWR1In0.LiV8bfKb2JUG2eIIxouXKebQpPFLXewO1BqoOD22xS4"
    url = f'{host}/{fileName}_download?'
    if start is not None: 
        url += f'archived_ts=gte.{start}'
        if end is not None: url += f'&archived_ts=lt.{end}'
    elif end is not None: url += f'archived_ts=lt.{end}'

    # Download the data using segmented download
    data = download_range(url,token,VERBOSE=True)

    # Sort the data
    data = sort(data)

    return data

# Save this data to a file to avoid having them in memory
def save_raw_data(data,filepath:str='./',buffer:int=1):
    """Save the raw data to a file in the computer

    Args:
        data (_type_): The raw data downloaded from NA server
        filepath (str, optional): The path that you want to save the file to. Defaults to './'.
        buffer (int, optional): The buffer number. Defaults to 1.

    Returns:
        string: The filename of the file.
    """
    # Create the filename
    timestamp   = '2022-NA-NAT' if len(data) == 0 else data[0]['archived_ts']
    date        = timestamp[0:timestamp.index('T')]
    filename    = filepath + f'light1-{date}-buff{buffer}.dat'

    # Load the file to write the output
    file = open(filename,'wb')

    # Append the data
    for row in data:
        # Convert the hexadecimal entry to bytes
        entry = bytes.fromhex(row['entry_data'][2:])
        file.write(entry)
    
    # Close the file
    file.close()

    # Return the filename if you need it
    return filename

# Convert from binary
def log_to_ascii(data,fileName:str=None):
    """Decode binary log file to ascii

    Args:
        data (dictionary): The dictionary obtained from the downloaded NA code
        fileName (str, optional): Filename to export the logfile to. If None then the file is not exported. Defaults to None.

    Returns:
        str: The decoded logfile as a string
    """
    # Store the full decoded text here
    full_text = ''

    # For every line in the logfile
    for entry in data:
        line =  bytes.fromhex(entry['entry_data'][2:]).decode("ASCII")
        full_text += line

    # If you need to store do so
    if fileName is not None: 
        file = open(fileName,'w')
        file.write(full_text)
        file.close()

    # Return the full text
    return full_text

# Parse a logfile and obtain metadata
def log_expand(filename:str=None,text:str=None):
    """Gets a logfile and decodes it to a list of commands. 
    If a text value is given then it decodes the text, if not, it then decodes the value from the filename

    Args:
        text (str, optional): The text of the logfile. Defaults to None.
        filename (str, optional): The filename of the file where the logfile is. Defaults to None.

    Raises:
        BaseException: If both parameters are left as None, then nothing happens. 

    Returns:
        decoded_logfile (list): List of lists. Each entry is a tuple with a command and a list for the outputs. 
    """

    # Do some argument processing:
    if filename is not None:
        # Load the logfile
        logfile = open(filename)

        # Load the lines
        loglines = logfile.readlines()

        # Close the file
        logfile.close()

    elif text is not None:
        loglines = text.split('\n')

    else: raise BaseException("Please enter input")

    # Add an SE0> line at the end if it doesn't exist
    if "SE0>" not in loglines[-1]: loglines.append("SE0>")

    # Decode the file
    # Find the indices of hte command lines
    commands_idx = [i for i,line in enumerate(loglines) if 'SE0>' in line]
    
    # Collect the outputs of the commands
    decoded_log = [[loglines[commands_idx[i]],loglines[commands_idx[i]+1:commands_idx[i+1]]] for i in range(len(commands_idx)-1)]

    # Return
    return decoded_log

# Parse custom command from satellite
def parse_custom_scenario(cmd:str):
    """Parses a custom scenario command message string to a dictionary of decoded hex values

    Args:
        cmd (str): Teh command message

    Returns:
        dict: The dictionary with outputs of all the relevant parameters set for the particular payload
    """
    # Store the data in a dictionary
    data = {}

    # Decode the information from the string
    data['hv']          = int(cmd[0:4],base=16)
    data['veto_hv']     = int(cmd[4:8],base=16)
    data['ch0_thresh']  = int(cmd[10:12]+cmd[8:10],base=16)
    data['ch1_thresh']  = int(cmd[14:16]+cmd[12:14],base=16)
    data['ch2_thresh']  = int(cmd[18:20]+cmd[16:18],base=16)
    data['ch3_thresh']  = int(cmd[22:24]+cmd[20:22],base=16)

    return data

# Obtain the metadata from a parsed logfile
def log_metadata(decoded_log:list):

    # metadata array initialization
    metadata = {
        'start_time':       None,
        'end_time':         None,
        'hv_SiPM':          -1,
        'hv_PMT':           -1,
        'hv_veto_SiPM':     -1,
        'hv_veto_PMT':      -1,
        'thresholds_SiPM':{
            'channel_0':    0,
            'channel_1':    0,
            'channel_2':    0,
            'channel_3':    0,
        },
        'thresholds_PMT':{
            'channel_0':    0,
            'channel_1':    0,
            'channel_2':    0,
            'channel_3':    0,
        },
        'custom_scenario_PMT': -1,
        'custom_scenario_SiPM': -1
    }

    # Get the command list
    commands = [row[0] for row in decoded_log]

    # Find the start and end of the data acquisition
    # Index of start and end timestamps:
    start = [i for i in range(len(commands)) if "rtc read" in commands[i]]
    if len(start) != 0: metadata['start_time']  = decoded_log[start[0] ][1][0][-21:-2]
    if len(start) >= 2: metadata['end_time']    = decoded_log[start[-1]][1][0][-21:-2]

    # Find the custom scenario commands for SiPM and PMT
    for num,payload in zip([12,13],['SiPM','PMT']):
        # Get all the commands with the custom scenario
        custom_commands = np.unique([commands[i] for i in range(len(commands)) if f"csp txrx {num} 9 3000" in commands[i]])
        
        # If there are any, decode them and replace
        if len(custom_commands) != 0: 
            message = custom_commands[0].split(' ')[-1][:-1]
            data    = parse_custom_scenario(message)

            # Update the decoded data to the metadata
            metadata['hv_'+payload]                         = data['hv']
            metadata['hv_veto_'+payload]                    = data['veto_hv']
            metadata['thresholds_'+payload]['channel_0']    = data['ch0_thresh']
            metadata['thresholds_'+payload]['channel_1']    = data['ch1_thresh']
            metadata['thresholds_'+payload]['channel_2']    = data['ch2_thresh']
            metadata['thresholds_'+payload]['channel_3']    = data['ch3_thresh']
            metadata['custom_scenario_'+payload]            = message
        

    # Return the metadata
    return metadata

# Download script packet
def download_data_packet(start:str=None,end:str=None,filepath:str='./',buffers=range(1,10)):
    """Download a packet of data from light-1 NA Server. This is the main library used.

    Args:
        start (str, optional): The start timestamp iso. Defaults to None.
        end (str, optional): The end timestmap in iso. Defaults to None.
        filepath (str, optional): The filepath to save everyhing. Defaults to './'.

    Returns:
        str: The madatadata of the filename
    """
    
    # Create a directory to store all this data
    if start is not None: filepath += 'light1-'+start[:start.index('T')]+'/'
    else: filepath += 'light1-data/'
    os.mkdir(filepath)

    # List that holds all the filenames
    filenames = []

    # First go ahead and download all the buffers
    for i in tqdm(buffers,desc='Downloading Buffer'):
        # Download the data of the buffer
        data    = download_time_delta(buffer=i,start=start,end=end)

        # Save the data of the buffer
        fname   = save_raw_data(data,filepath=filepath,buffer=i)
        filenames.append(fname)

    # Download the script log
    log         = download_log(start=start,end=end)
    if start is not None: log = log_to_ascii(log,fileName=filepath+'light1-'+start[:start.index('T')]+'-se-log.txt')
    else: log = log_to_ascii(log,fileName=filepath+'light1-se-log.txt')
    decoded_log = log_expand(text=log)

    # Extract the metadata from the logfile
    metadata = log_metadata(decoded_log=decoded_log)

    # Save the datafile as a json on the same directory
    with open(filepath + "metadata.json","w") as meta_file: json.dump(metadata,meta_file,indent=4)

    return metadata

# Parse a command
def desc_finder(line:str,cmdlist,outputs,i,time,failed_idx):
    """Parse a command and return its status and description

    Args:
        line (str): The string of the command
        cmdlist (pd.DataFrame): pandas data frame with the commands and their equivalent messages
        outputs (_type_): _description_
        i (_type_): _description_
        time (_type_): _description_
        failed_idx (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Get the description
    status = 1
    splt = line.split(' ')

    # define end of log file
    if splt[-1] == 'SE0>':
        # desc = 'LOG END'
        desc = [-1,17]

    # define commands from the command file
    elif 'txrx' in splt[1]:
        node,port,msg = int(splt[2]),int(splt[3]),str(splt[5])

        index = cmdlist.loc[(cmdlist['NODE']==node) & (cmdlist['PORT']==port) & (cmdlist['Message'].str.startswith(msg)),['ID_COMMAND_Proposed','ID_in_Graph']]
        
        # include the power shutdown
        if node == 4:
            index = cmdlist.loc[(cmdlist['NODE']==node) & (cmdlist['PORT']==port),['ID_COMMAND_Proposed','ID_in_Graph']]

        # include the custom scenario
        if port == 9:
            index = cmdlist.loc[(cmdlist['NODE']==node) & (cmdlist['PORT']==port),['ID_COMMAND_Proposed','ID_in_Graph']]
        

        #If index did not find anything
        if len(index) == 0:
            desc = []
            
        # if command found in command list
        else:
            desc = list(index.to_numpy()[0])
        

        if i in [fid for fid in failed_idx]:
            time = time + (float(splt[4])/1000)
            status = -1
        

    elif splt[1] == 'delay':
        desc = []

    elif splt[1] == 'delayuntil':
        time = float(splt[2])
        desc = []

    elif 'read' in splt[1]:
        time = float(outputs[i][0].split(' ')[3])
        desc = []

    else:
        desc = []
    
    return desc,time,status



# Decode a logfile
def decode_log(filename:str="../../../Data/Logs/light1-SD-1016-se-log.txt"):
    """Take a log file and parse its commands to uncover thier timestamps

    Args:
        filename (str, optional): The filepath and filename of the log file. Defaults to "../../../Data/Logs/light1-SD-1016-se-log.txt".

    Returns:
        _type_: Log, commands, outputs, description, failed_idx, loglines_array
    """

    # Load the logfile
    logfile = open(filename)
    cmdlist = pd.read_csv("command_list.csv")

    # Load the lines
    loglines = logfile.readlines()

    # Close the file
    logfile.close()

    # Create an array with the lines
    commands    = []
    outputs     = []
    idx         = []
    description = []
    times       = []
    fails       = []
    failed_idx  = []

    # Get commands and their indices
    for i, line in enumerate(loglines):
        if 'SE0>' in line:
            commands.append(line)
            idx.append(i)

    # Check if the last command was empty
    if loglines[-1] != 'SE0>':
        commands.append('SE0>')
        idx.append(len(loglines))

    # Get the command output
    for i in range(len(idx)-1):
        out = []
        if 'SE0>#' not in commands[i]:
            for j in range(idx[i]+1,idx[i+1]):
                out.append(loglines[j])
        
        outputs.append(out)

    # Create the dictionary
    log = [[command,output] for command,output in zip(commands,outputs)]

    # Check which commands executed correcly
  

    # Find the commands that failed
    for i in range(len(commands)-1):
        for output in log[i][1]:
            if 'FAIL' in output:
                failed_idx.append(i)


    # Initialize time at 0s
    time = 0
    
    # Give commands and use the finder function to pull the description, time ran, and if the command failed.
    for k, id in enumerate(idx):
        c = commands[k]
        cmd = c.split('\n')[0]
        desc,time,failed = desc_finder(cmd,cmdlist,outputs,k,time,failed_idx)

        if len(desc) != 0:
            description.append(desc)
            times.append(time)
            fails.append(failed)
    tempname = 'templog.csv'
    fileDir = os.path.join('Defined_Logs')

    if not os.path.exists(fileDir):
        os.makedirs(fileDir)
    
    filePath = os.path.join(fileDir, tempname)

    #Name the headers of the CSV File
    header = ['status','time','description','ID_in_Graph']

    loglines_array = []

    #Clear the file and write from scratch
    with open(filePath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for i in range(len(description)):
            g = (fails[i], times[i-1], description[i][0],description[i][1])
            loglines_array.append(g)
            writer.writerow(g)

    # Return everything else
    return log, commands[:-1], outputs, description, failed_idx, loglines_array
