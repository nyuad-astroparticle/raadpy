from tqdm import tqdm
import numpy as np
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
BYTE            = 8                        # Byte length

NONVETO_STRUCT  = {
    'channel'       : 2,
    'adc_counts'    : 10,
    'stimestamp'    : 36,
}

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

def find_pairs(diffs,MEAN,STD):
    """Helper function to find the pairs of bit flips occuring in a timestring, Use invert_flips instead!
    """
    candidates  = np.array([])
    idxs        = np.array([],dtype=int)
    round_msb = lambda data,BITS: get_msb((3*abs(data)).astype(int) >> 1,BITS)*np.sign(data)

    pairs = []    
    for i,d in enumerate(round_msb(diffs,50)):
        # If this is a candidate for a thing 
        if d < -MEAN - 3*STD:
            candidates = np.append(candidates,[d])
            idxs       = np.append(idxs      ,[i])
        
        if len(candidates) > 0:
            # candidates += d
            idx = np.where(abs(candidates + d) == 0)[0]
            if len(idx) > 0:
                index = idx[0]
                pairs.append((idxs[index],i))
                candidates = np.delete(candidates,[index,int(-1)],axis=0)
                idxs       = np.delete(idxs      ,[index,int(-1)],axis=0)

    return np.array(pairs)

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
    pairs = find_pairs(timestamp_deltas,MEAN,STD)
    
    # For each bit flip region
    for pair in pairs:
        ADD = 0
        # For each point within the region
        for i in range(*pair):
            # Calculate a correction and apply it
            ADD += correct_bit(int(timestamp_deltas[i]),MEAN,BITS=BITS) - timestamp_deltas[i]
            timestamp[i+1] += ADD

    return timestamp
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
def get_bit_idx(n:int):
    return BYTE - 1 - n%BYTE + (n//BYTE) * BYTE
def get_msb(x:int,BITS=48):
    for i in range(BITS):
        x |= x >> int(2**i)
    return (x+1) >> 1

def get_dict(filename:str,struct=ORBIT_STRUCT,condition:str=None,MAX=None,STUPID:bool=False,VERIFY=False,threshold=5e-5):
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
    # Initialize the dictionary
    data = dict(zip(struct.keys(),[ [] for _ in range(len(ORBIT_STRUCT.keys()))]))
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
                data[name].append(get_bits(bits_read,length,event,STUPID=STUPID))
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
        # Split to channels
        channels, cnt = split_channels(data,struct)
        
        # Apply correction to each channel
        for channel in channels: channel['stimestamp'] = invert_flips(channel['stimestamp'],struct['stimestamp'])

        # Put it back together
        for i, channel in enumerate(channels):
            for j, time in enumerate(channel['stimestamp']):
                data['stimestamp'][cnt[i][j]] = time
    
    # Return the dictionary
    return data
