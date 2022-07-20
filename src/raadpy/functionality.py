#############################
#     RAAD Functionality    #
#############################

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
def get_dict(filename:str,struct=ORBIT_STRUCT,condition:str=None,MAX=None,STUPID:bool=False):
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
    host=HOST
    token=TOKEN

    # Create a rest request
    rest = RestOperations(f'{host}/{fileName}_download?file_ver=eq.{file_ver}', authType = 'token', token = token)
   
    # Download the data using the request
    data = rest.SendGetReq()

    # Sort the data
    data = sort(data)

    return data

# Download data based on time range
def download_time_delta(buffer:int = 1, start:str='2022-06-01T00:00:00', end:str='2022-06-07T00:00:00'):
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
    host=HOST
    token=TOKEN
    
    # Create a rest request
    rest = RestOperations(f'{host}/{fileName}_download?archived_ts=gte.{start}&archived_ts=lt.{end}', authType = 'token', token = token)

    # Download the data using the request
    data = rest.SendGetReq()

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
    timestamp   = data[0]['archived_ts']
    date        = timestamp[0:timestamp.index('T')]
    filename    = filepath + f'Light1_{date}_Buff{buffer}.dat'

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