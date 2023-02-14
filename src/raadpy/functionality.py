#############################
#     RAAD Functionality    #
#############################

from .core import *
from .rparray import array
from .event import *

# Split the dataset in channels
def split_channels(data,struct=NONVETO_STRUCT):
    """Split the data based on their channels

    Args:
        data (_type_): Buffer data
        struct (_type_, optional): Structure to decode them as. Defaults to NONVETO_STRUCT.

    Returns:
        channels: List of lists for all the channels
    """
    # Split the data based on their channels
    channels    = []
    idxs        = []
    for channel in np.unique(data['channel']):
        idx         = np.where(data['channel'] == channel)[0]
        idxs.append(idx.copy())
        channels.append(dict(zip(struct.keys(),[arr[idx] for arr in data.values()])))
    
    return channels,idxs

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
    # If hardware accelleration is not available use numpy instead
    if not CUPY_AVAILABLE:cp = np

    # If we are given an array of TGFs
    if type(tgf) == array:
        # Create a list to output the lighning arrays for each event
        tgf = cp.asarray(tgf)
        lights = cp.array([])

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
        timestamps = cp.asarray(lightnings.get_timestamps())

        # find the indices where the timedifference is less than threshold
        # idx = [i for i,time in enumerate(timestamps) if abs(time - tgf_time) < threshold]
        idx = cp.asarray((abs(timestamps -tgf_time.to_value(format="unix")) < threshold.to_value(format="sec")).nonzero())
        idx = idx.get()[0]
        # Get the appropriate subarray
        return array(lightnings[idx.to_list()])

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


# Recursively keep removing the Most Significant Bit, until you're below the mean threshold
def correct_bit(x:int, MEAN:float, BITS:int=48):
    """Recursively Correct a number for bit flips, by removing the most significant bit until it's below a threshold

    Args:
        x (int): An integer to correct
        MEAN (float): The value to reach
        BITS (int, optional): The maximum number of bits that the number can have. Defaults to 48.

    Returns:
        x (int): The corrected number
    """
    if abs(x) <= MEAN: 
        return x
    if x > 0: return correct_bit(x - get_msb(x,BITS),MEAN,BITS)
    else: return correct_bit(x + get_msb(abs(x),BITS),MEAN,BITS)


# Find pairs
def find_pairs(diffs,MEAN,STD,BITS:int=50):
    """Helper function to find the pairs of bit flips occuring in a timestring, Use invert_flips instead!
    """
    candidates  = np.array([])
    idxs        = np.array([],dtype=int)
    round_msb = lambda data,BITS: get_msb((3*abs(data)).astype(int) >> 1,BITS)*np.sign(data)

    pairs = []    
    for i,d in enumerate(tqdm(round_msb(diffs,BITS),desc='Bit Flipper')):
        # If this is a candidate for a thing 
        if abs(d) > MEAN + 2*STD:
            candidates = np.append(candidates,[d])
            idxs       = np.append(idxs      ,[i])

        if len(candidates) > 0:
            # candidates += d
            idx = np.where(abs(candidates + d) == 0)[0]
            if len(idx) > 0:
                for index in idx:
                    if abs(i-idxs[index]) < 5000:
                        pairs.append((idxs[index],i))
                        candidates = np.delete(candidates,[index,int(-1)],axis=0)
                        idxs       = np.delete(idxs      ,[index,int(-1)],axis=0)
                        break

    return np.array(pairs)


def invert_flips(timestamp:np.array,BITS:int=NONVETO_STRUCT['stimestamp']):
    """Given a list of timestamps, find and correct the bit flips occured.

    Args:
        timestamp (np.array): The array of timestamps
        BITS (dict, optional): The number of bits in the timestamp variable. Defaults to NONVETO_STRUCT['stimestamp'] = 48.

    Returns:
        timestamp (np.array): The corrected timestamp
    """
    # Get the gradient of the timestamp
    timestamp_deltas = timestamp[1:] - timestamp[:-1]

    # Get Mean and standard deviation
    MEAN = np.mean(abs(timestamp_deltas))
    STD  = np.std(abs(timestamp_deltas))

    # Identify the pairs of points where you get bit flips
    pairs = find_pairs(timestamp_deltas,MEAN,STD,BITS)

    # For each bit flip region
    for pair in pairs:
        ADD = 0
        # For each point within the region
        for i in range(*pair):
            # Calculate a correction and apply it
            ADD += correct_bit(int(timestamp_deltas[i]),MEAN,BITS=BITS) - timestamp_deltas[i]
            timestamp[i+1] += ADD

    return timestamp

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

def get_dict(filename:str,struct=ORBIT_STRUCT,condition:str=None,MAX=None,STUPID:bool=False,VERIFY=False,threshold=5e-5,LAST:int=None,min_streak:int=-1):
    """Decode the data of a buffer with a given structure into a dictionary

    Args:
        filename (str): The filename where the buffer is
        struct (_type_, optional): The structure of the bits of the buffer represented in a dictionary. Defaults to ORBIT_STRUCT.
        condition (str, optional): If you want you can add a condition such as data['id_bit']==1 to filter the data as they're being loaded. Defaults to None.
        MAX (_type_, optional): Maximum number of lines to read, if None then read all of them. Defaults to None.
        STUPID (bool, optional): Should be set to True if you are reading VETO and NONVETO. Defaults to False.
        VERIFY (bool, optional): Set to True to process error correction automatically, such as filtering the 2-byte error, and correcting for bit flips. Defaults to False
        threshold (float,optional): The difference between two points in the timestamp that we can consider faulty as a fraction of the maximum number that the integer field can store. If threshold > 1 then it is considered as an absolute threhsold. Only used if VERIFY=True. Defaults to 5e-5

    Returns:
        data (dict): Dictionary with the decoded arrays of measurements
    """
    # Read the raw data
    file = open(filename,'rb')  # Open the file in read binary mode
    raw = file.read()           # Read all the file
    file.close()                # Close the file
    
    # Number of bytes per line
    bytes_per_line  = sum(list(struct.values()))//8
    length          = len(raw)//bytes_per_line
    if MAX is None or MAX > length: MAX = length
    if LAST is not None: MAX = LAST
    
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
    
    # Current byte index in the file
    curr = 0 if LAST is None else len(raw) - LAST * bytes_per_line

    # Initialize the dictionary
    data = dict(zip(struct.keys(),[ [0]*MAX for _ in range(len(struct.keys()))]))

    with tqdm(total=MAX,desc='Line: ', miniters=10) as pbar:
    
        # Counter for streaks of zero
        zero_streak = 0
        last_curr   = 0
        last_i      = 0
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
                # data[name].append(get_bits(bits_read,length,event,STUPID=STUPID))
                data[name][i] = get_bits(bits_read,length,event,STUPID=STUPID)
                bits_read += length

            # Verify the datum makes sense
            if VERIFY:
                # Update the zero streak counter
                if min_streak >= 0:
                    if data['channel'][i] == 0: zero_streak += 1
                    elif zero_streak > min_streak:
                        tqdm.write(f'ZERO COUNTS TRIGGERED AT {i} with zero_streak = {zero_streak}, curr = {curr}')
                        # i          -= zero_streak - 1
                        # curr       -= -2 + bytes_per_line * (zero_streak)
                        i           = last_i - 10
                        curr        = last_curr + 1 - 10*bytes_per_line
                        last_curr   = curr
                        update      = 0.1
                        pbar.colour = '#ffff00'
                        pbar.total += 2
                        zero_streak = 0
                        tqdm.write(f'\t     RETRYING AT {i} with zero_streak = {zero_streak}, curr = {curr}')
                        continue
                    else: 
                        zero_streak = 0
                        last_curr   = curr
                        last_i      = i

                # If there are more than two datapoints in the timestamp
                if i>=1:#len(data['stimestamp'])>=2:
                    # If the difference between the last two timestmaps is absurd
                    if (data['stimestamp'][i] - data['stimestamp'][i-1] > THRESHOLD) or (THRESHOLD*0.05 < data['stimestamp'][i-1] -  data['stimestamp'][i] < THRESHOLD*0.3):# or data['stimestamp'][i] > 2*THRESHOLD:# or (data['adc_counts'][-1] <= 3):
                    # if data['stimestamp'][-1] - data['stimestamp'][-2] > THRESHOLD:# or (data['adc_counts'][-1] <= 3):
                        # tqdm.write(f"{curr}: {data['stimestamp'][-1]} - {data['stimestamp'][-2]} = {abs(data['stimestamp'][-1] - data['stimestamp'][-2])} > {THRESHOLD} {abs(data['stimestamp'][-1] - data['stimestamp'][-2]) > THRESHOLD}")
                        # remove the previous datapoint
                        # for key in data.keys():
                        #     # print(data[key][-1])
                        #     data[key] = data[key][:-1]
                        # Move forward by two bytes
                        curr   -= - 1 + bytes_per_line
                        i      -= 1
                        update  = 1.1
                        pbar.colour = '#ff0000'
                        pbar.total += 1

                # elif data['stimestamp'][-2] - data['stimestamp'][-1] > THRESHOLD:

                elif i==0 and (data['stimestamp'][0] > THRESHOLD or data['adc_counts'][0] <= 3) :
                # elif len(data['stimestamp'])==1 and (data['stimestamp'][0] > THRESHOLD or data['adc_counts'][0] <= 3) :
                    # for key in data.keys():
                    #     # print(data[key][-1])
                    #     data[key] = data[key][:-1]
                    # Move forward by two bytes
                    # tqdm.write(f"{curr}: {data['stimestamp'][-1]} - {data['stimestamp'][-2]} = {abs(data['stimestamp'][-1] - data['stimestamp'][-2])} > {THRESHOLD} \n\t{data['adc_counts'][0]}")
                    curr   -= - 1 + bytes_per_line
                    i      -= 1
                    update  = 1.1
                    pbar.colour = '#ff0000'
                    pbar.total += 1
                    

            # Update reader position
            curr    += bytes_per_line
            i       += 1
            # print(curr,update)
            pbar.update(np.floor(update))
            if update == 1: pbar.colour = '#0000ff'

    for name, value in data.items():
        data[name] = np.array(value) 
    # If you want to filter, then apply the filter to the loaded data directly
    if condition is not None:
        try:
            idx     = np.where(eval(condition))[0]
            data    = dict(zip(struct.keys(),[arr[idx] for arr in data.values()]))
        except:
            print(bcolors.WARNING+'WARNING!' + bcolors.ENDC +' Condition ' + condition + ' is not valid for the dataset you requested. The data returned will not be filtered')
    # Specific loading changes
    if 'temperature' in struct.keys():
        data['temperature'] = [i - 55 for i in data['temperature']]
        

    # If we can do a bit flip verification perform it
    if VERIFY:
        # # Split to channels
        # channels, cnt = split_channels(data,struct)

        # # Apply correction to each channel
        # for channel in channels: channel['stimestamp'] = invert_flips(channel['stimestamp'],struct['stimestamp'])

        # # Put it back together
        # for i, channel in enumerate(channels):
        #     for j, time in enumerate(channel['stimestamp']):
        #         data['stimestamp'][cnt[i][j]] = time

        data['stimestamp'] = invert_flips(data['stimestamp'],struct['stimestamp'])
    
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
def download_range(url:str,token,limit:int=5000,VERBOSE:bool=False,clear=clear):
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
        append = (f'&limit={limit}&seq_nr=gt.{seq}' if url[-1] != '?' else f'limit={limit}&seq_nr=gt.{seq}') if limit >= 0 else ''
        rest = RestOperations(url+append, authType = 'token', token = token)
       
        # Download the data
        last_data   = rest.SendGetReq()
        data        += last_data

        # If there are no more data exit
        if len(last_data) < limit or limit == -1: # or seq == max([datum['seq_nr'] for datum in data]):
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
        start (str, optional): String with iso date to start. Defaults to None.
        end (str, optional): String with iso date to end. Defaults to None.

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
def log_expand(filename:str=None,text:str=None,remove_comments:bool=False):
    """Gets a logfile and decodes it to a list of commands. 
    If a text value is given then it decodes the text, if not, it then decodes the value from the filename

    Args:
        text (str, optional): The text of the logfile. Defaults to None.
        filename (str, optional): The filename of the file where the logfile is. Defaults to None.

    Raises:
        BaseException: If both parameters are left as None, then nothing happens. 

    Returns:
        decoded_logfile (list): List of dictionaries. Each entry is a tuple with a command and a list for the outputs. 
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
    # Find the indices of the command lines
    commands_idx = [i for i,line in enumerate(loglines) if ('SE0>' in line) and (('#' not in line) or not remove_comments)]

    # Collect the outputs of the commands
    decoded_log = [{
        'command':loglines[commands_idx[i]],
        'output' :loglines[commands_idx[i]+1:commands_idx[i+1]],
        'index'  :commands_idx[i]
        } for i in range(len(commands_idx)-1)]

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
    commands = [row['command'] for row in decoded_log]

    # Find the start and end of the data acquisition
    # Index of start and end timestamps:
    start = [i for i in range(len(commands)) if "rtc read" in commands[i]]
    if len(start) != 0: metadata['start_time']  = decoded_log[start[0] ]['output'][0][-21:-2]
    if len(start) >= 2: metadata['end_time']    = decoded_log[start[-1]]['output'][0][-21:-2]

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

def log_line_timestamp(logline:list,time:float=0):
    """Given a log file line as the lines decoded by log_expand, and a time
    return the time increment for this line

    Args:
        logline (list): The element of the decoded logfile corresponding to a line
        time (float): The time up to now

    Returns:
        time (float): Time increment after the execution of the command
    """

    splt = logline['command'].split(' ')                            # Split the cmnd line (logline[0]) by the spaces 
    if "SE0>\n" not in splt[0] and len(splt) > 1:                   # If the cmnd is not the end of the log file: 
        
        try:
            if "FAIL\n" in logline['output']:                       # if it was a payload cmnd and it failed (there will be delays)
                if "txrx" in splt[1]:time + float(splt[4])/1000     # take the time a payload cmnd takes to be excuted 

            elif "read" in splt[1]:                                 # If cmnd is read time
                time = float(logline['output'][0].split(" ")[3])   # Replace time with the time read

            elif splt[1] == "delay":                                # If the cmnd is a delay
                time = time + (float(splt[2])/1000)                 # Add time delay to previous time

            elif splt[1] == "delayuntil":                           # If cmnd is delayuntil 
                time = float(splt[2])                               # Replace time by new time
        
        except ValueError:
            print('Found weird command\n', logline)
        
        except IndexError:
            print("Index Error in command\n",logline)

    return time

# Now we can define an error correction pass
def find_closest(x,array,return_idx:bool=False):
    """Helper function that returns the two closest values to x in a list

    Args:
        x (numeric): The value to search for
        array (iterable): The list to search in

    Returns:
        (a,b) (tuple): The two values closest to x that can be found in the list. If x is in the list, b=x and a<b.
    """
    # Binary search
    s   = 0
    e   = len(array)-1
    mid = (s + e)//2
    prevmid = -float('inf')

    # Handle edge cases
    if x <= array[s]: return (-float('inf'),array[s]) if not return_idx else [(-float('inf'),array[s]),(-float('inf'),s)]
    if x >= array[e]: return (array[e],float('inf'))  if not return_idx else [(array[e],float('inf')), (e,float('inf'))]

    # Flag that will update condition to exit
    while prevmid != mid:
        if x == array[mid]: return (array[mid-1],array[mid]) if not return_idx else [(array[mid-1],array[mid]),(mid-1,mid)]
        if x < array[mid]:  e = mid
        if x > array[mid]:  s = mid

        prevmid = mid
        mid     = (s+e)//2

    if (s==e): return (array[s],array[s+1]) if not return_idx else [(array[s],array[s+1]),(s,s+1)]
    return (array[s],array[e]) if not return_idx else [(array[s],array[e]),(s,e)]



def reorder_log(logfile:list):
    """Reorder a logfile based on it's timestamps

    Args:
        logfile (list): The logifile after having a timestamp field added

    Returns:
        logfile (list): Corrected logfile in time
    """

    # Get a list of timestamps
    timestamps = np.array([line['timestamp'] for line in logfile])

    # Get the indices where the time is not increasing
    idx_flip = np.where(timestamps[1:] - timestamps[:-1] < 0)[0] + 1

    # Get he indices of the rtc reads
    idx_read = np.append([0],[np.where('SE0>rtc read\n' == np.array([line['command'] for line in logfile]))[0]])

    # Add the difference to the regions of interest
    for i in idx_flip:
        # Calculate the time difference
        diff = logfile[i-1]['timestamp'] - logfile[i]['timestamp']
        
        # Find the closest two rtc read timestamps
        _, closest_rtc = find_closest(i,idx_read)

        # Correct the timestamps in this region
        while (logfile[closest_rtc-1]['timestamp'] - logfile[closest_rtc]['timestamp']) > 0:
            logfile[closest_rtc-1]['timestamp'] -= diff
            closest_rtc -= 1


    return logfile


def log_with_timestamp(logfile:list,reorder=True):                        
    """Given a log file from the function log_expand
    return a list of log lines with the time increment of each line

    Args:
        logfile (list): a list of cubesat commands

    Returns:
        log_timestamp_list (list): a list of cubesat commands, each command with its timestamp
    """
    log_timestamp_list=[]                               # List to be returned later (cmnd, output, timestamp)
    time = 0                                            # Set initial time to zero, this step will not be needed
    for logline in logfile:
        time = log_line_timestamp(logline,time)
        log_timestamp_list.append ({
            'command':      logline['command'],
            'output':       logline['output'],
            'index':        logline['index'],
            'timestamp':    time
        })

    return reorder_log(log_timestamp_list) if reorder else log_timestamp_list


def get_cmd_number(cmdline:str):
    """Get the command number of a specific command line from the log file.

    Args:
        cmdline (str): The command lines as given by the log files. Example: "SE0>csp txrx 12 8 3000 0F".

    Returns:
        value (int): The number of the command on the buff 1 graphs. The value will be -1 for nonpayload cmds.
    """
    
    value = -1                          # Initialize the value to be -1 so that if the function faces commands that are not in the CMND_LIST it will give it a -1 value
    split_cmdline = cmdline.split(" ")  # Split the cmd by the spaces in between
    if len(split_cmdline) > 1 and 'txrx' in split_cmdline[1]: # If the cmd is a payload cmd, is has txrx
        try:
            if float(split_cmdline[2]) == 4:                      # If the cmd belongs to emergency poweroff then the message is not needed (message is split_cmdline[5])
                value = CMND_LIST[split_cmdline[2]][split_cmdline[3].strip()]['number_in_graph'] # Grab the cmd number from the CMND_LIST based on the node and port of the cmd ([split_cmdline[2]][split_cmdline[3]])
            else:
                value = CMND_LIST[split_cmdline[2]][split_cmdline[3].strip()][split_cmdline[5].strip()]['number_in_graph'] #If not an emergency poweroff use node, port, and message to find the cmd in the CMND_LIST
        except KeyError:
            # print(cmdline)
            value = 14

    return value 
        


def send_sql_query_over_ssh(query:str):
    """Receives SQL query and sends it to the raad@arneodolab.abudhabi.nyu.edu SQL server.

    Args:
        query (str): SQL Formatted Query

    Returns:
        data (pandas.dataframe): Dataframe of the data requested as formatted on the server.
    """

    # Set up ssh tunnel
    # tunnel = paramiko.SSHClient()
    # tunnel.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # tunnel.connect('arneodolab.abudhabi.nyu.edu', 22,username='raad',password='nyuad123$',allow_agent=False,look_for_keys=False)
    tunnel  = SSHTunnelForwarder(('arneodolab.abudhabi.nyu.edu', 22), ssh_password="nyuad123$", ssh_username="raad", remote_bind_address=('127.0.0.1', 3306), allow_agent=False,)
    tunnel.start()

    # Connect to PYSQl Host
    conn    = pymysql.connect(host='127.0.0.1', user='raad', passwd="nyuad123$", port=tunnel.local_bind_port)
    data    = pd.read_sql_query(query, conn)    # Request Data
    tunnel.close()                              # Close tunnel
    
    return data


def get_light1_position(starttime=None, endtime:Time=None, n_events:int=None, SHORT_TIME:float=10):
    """Give me a start and end time in astropy objects, I give you light-1 position. 
    I can also interpolate if you give me anumber of events

    Args:
        starttime (Time): Start time, or single event time
        endtime (Time, optional): End time. If left None, it assumes single position. Defaults to None.
        n_events (int, optional): Number of points in case we interpolate. Defaults to None.
        SHORT_TIME (float, optional): If you want a single event, give us some wiggle room to look around for it. Defaults to 10.

    Raises:
        Exception: Start Time and End Time are not astropy Time objects
    
    Returns:
        locs (rp.array): Raadpy array with the locations of the satellite in this interval.
    """
    # Input processing on start time
    if type(starttime) != Time:
        try:
            starttime = Time(starttime)
        except:
            raise Exception("Given timestamp is not an astropy Time object")       

    # If endtime is not given, it means we want only one event, so we will interpolate around SHORT_TIME
    if endtime is None:
        starttime   -= TimeDelta(SHORT_TIME,format='sec')
        endtime      = starttime + TimeDelta(2*SHORT_TIME,format='sec')
        n_events    = -1
    else: 
        try:
            endtime     = Time(endtime)
        except:
            raise Exception("Start time and End time are not astropy Time object")

    # Request the data from sql for this time period
    data =(send_sql_query_over_ssh("SELECT * FROM `LIGHT-1_Position_Data`.PositionData WHERE `Time (ModJDate)` BETWEEN " + str(starttime.to_value("mjd")) + " AND " + str(endtime.to_value("mjd")) + ";"))
    print("SELECT * FROM `LIGHT-1_Position_Data`.PositionData WHERE `Time (ModJDate)` BETWEEN " + str(starttime.to_value("mjd")) + " AND " + str(endtime.to_value("mjd")) + ";")
    
    # Get the data from the dataframe
    latitudes   = [i for i in data['Lat (deg)']]
    longitudes  = [i for i in data['Lon (deg)']]
    times       = [i for i in data['Time (ModJDate)']]
    
    # Interpolation if needed
    if n_events is not None:
        times_tmp  = np.linspace(starttime.to_value(format="mjd"), endtime.to_value(format="mjd"), n_events) if n_events > 0 else np.array([(starttime.mjd+endtime.mjd)/2])
        latitudes  = np.interp(times_tmp, times, latitudes)
        longitudes = np.interp(times_tmp, times, longitudes)
        times = times_tmp
    
    # Create rp.array
    locs = array(event_type="location")
    for i in range(len(times)):
        locs.append(event(
            timestamp   = Time(times[i], format="mjd"),
            longitude   = in_range(float(longitudes[i])),
            latitude    = float(latitudes[i]),
            detector_id = 'NA',
            mission     = 'NanoAvionics',
            time_format = "mjd",
            event_type  = "cubesat-location"
        ))

    return(locs)

def get_lightning_time_ranges(start_time, end_time):
    """Generates a list of time ranges for each day encompassed within start_time and end_time

    Args:
        start_time (Time): start time
        end_time (Time): end time
    
    Returns:
        return_list(list): A list of time ranges. The first element is the start time, the second element is the end time, and the third element is the day corresponding to the time range. If the first element is 0, the start time is at the start of the day. If the second element is zero, the end time is the end of the day.
    """
    # Calculating the number of days between the given start and end times by rounding them down and subtracting start time from end time
    rounded_start = start_time.to_datetime().replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    rounded_end = end_time.to_datetime().replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    days = (rounded_end - rounded_start).days

    # Creating start and end days using the start and end times    
    full_start_day = start_time.to_value("isot")
    start_day = full_start_day[:4] + full_start_day[5:7] + full_start_day[8:10]
    
    full_end_day = end_time.to_value("isot")
    end_day = full_end_day[:4] + full_end_day[5:7] + full_end_day[8:10]
    
    # Creating a list of time ranges which is to be returned
    return_list = []
    if days ==0:
        # If the start and end time are within the same day, simply add the start and end times to the start day element in return_list
        return_list.append([start_time, end_time, start_day])
    else:
        # If start and end time are on different days, then first of all add an element corresponding to the start time and start day
        return_list.append([start_time, 0, start_day])
        
        # Create a counter variable to keep track of which day's element is being added to retrun_list
        current_day = dt.datetime(int(start_day[0:4]), int(start_day[4:6]), int(start_day[6:]))

        for i in range(days-1):
            # For each day in between start day and end day, increment the current day by one day and add an element that corresponds to all the timestamps in that day to return_list
            current_day = current_day + dt.timedelta(days=1)
            temp_day = str(current_day)
            return_list.append([0, 0, Time(temp_day)])
        
        # Finally add an element to return_list correspond to the end time and end day
        return_list.append([0, end_time, end_day])
    
    return(return_list)

def get_lightning_strikes(starttime:Time, endtime:Time):
    """Downloads lightning strikes within the given start and end times and loads them into a rp.array

    Args:
        starttime (Time): Start time 
        endtime (Time): End time

    Raises:
        Exception: Start time and end time must be astropy.Time objects

    Returns:
        locs (rp.array): raadpy array consisting of lightning strikes between the given start and end times
    """

    # Input Processing on start and end times
    if (type(starttime) != Time) or (type(endtime) != Time):
        try:
            starttime = Time(starttime)
            endtime = Time(endtime)
        except:
            raise Exception("Start time and end time must be astropy.Time objects")

    # converting the give start and end times to a list of time ranges       
    time_ranges = get_lightning_time_ranges(starttime, endtime) 

    # Initialising the rp.array that will be returned
    locs = array(event_type="en-lightning") 

    # Requesting the lightning data for each time range
    for i in tqdm(time_ranges):
        if (i[0] != 0) & (i[1] != 0):
            data = send_sql_query_over_ssh(f"SELECT timestamp,latitude,longitude, peakcurrent FROM LightningData.`{i[2]}` WHERE timestamp BETWEEN ' {i[0]}' AND ' {i[1]}';")

        if (i[0] != 0) & (i[1] == 0):
            data = send_sql_query_over_ssh(f"SELECT timestamp, latitude, longitude, peakcurrent FROM LightningData.`{i[2]}` WHERE timestamp >= ' {i[0]}'")

        if (i[0] == 0) & (i[1] != 0):        
            data = send_sql_query_over_ssh(f"SELECT timestamp, latitude, longitude, peakcurrent FROM LightningData.`{i[2]}` WHERE timestamp <= ' {i[1]}'")

        times = [i for i in data['timestamp']]
        latitudes = [i for i in data['latitude']]
        longitudes = [i for i in data['longitude']]
        currents = [i for i in data['peakcurrent']]

        # Creating a rp.event object for each lightning strike and adding that to locs
        for i in tqdm(range(len(times)),leave=False,desc=f'Day:',colour='#57068c'):
            locs.append(event(
                timestamp   = Time(times[i].strip(), format="isot"),
                longitude   = longitudes[i],
                latitude    = latitudes[i],
                detector_id = "EN",
                mission     = "Earth Networks",
                time_format = "isot",
                event_type  = "en-lightning",
                property    = float(currents[i])
            ))

    return(locs)
    
def parse_command(cmd:str):
    '''
    Parses a string payload command from the logfile and returns a human readable version for it

    Input:
        - cmd (str): The command string from the logfile
    
    Output:
        - str: Human readable version of command
    '''
    if 'txrx' in cmd:
        # Decompose the command
        e           = cmd.strip().split(' ')
        elements    = [e[2],e[3]]
        if e[2] != '4': elements.append(e[-1])

        # Read the nested dictionary
        d = CMND_LIST
        for el in elements:
            d = d[el]

        # Get the output
        return d['description']
    
    else:
        return 'Not a Payload Command'