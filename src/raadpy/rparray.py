#############################
#     RAAD ARRAY class      #
#############################

from .core import *
from .event import *

# Since we will be working with tgf arrays so much we will create
# Another class called TGF Array that has methods to handle an array of TGFs
class array:
    event_types = ['fermi','light-1','lightning','location']

    # Constructor
    def __init__(self,events=None,filename='',event_type=event_types[0]):
        """Array of events

        Args:
            events (_type_, optional): A list of event objects. If None the array is initialized with other methods. Defaults to None.
            filename (str, optional): Filenmae from which to draw the events from. If '' the array is initialized with other methods. Defaults to ''.
            event_type (_type_, optional): What type of event is it? Certain keywords can be used to define special events for imports ('fermi','light-1','lightning','location'). Defaults to event_types[0].
        """
        # Add the tgfs as an array
        self.events = [] if events is None else events

        # if you add a filename, add the apppend the tgfs from that filename
        if filename != '':
            self.from_file(filename=filename,event_type=event_type,append=True)


    # Method to append
    def append(self,ev):
        """Appends an event to the array

        Args:
            ev (event): the event to append
        """
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
        """Get a numpy array of the coordinates of the events in the list

        Returns:
            Coordinates (np.array): List of tuples of type (long,lat) for all the events in the array
        """
        coords = []
        for event in self.events:
            coords.append([event.longitude,event.latitude])
        
        return np.array(coords)

    # Get array of timestamps
    def get_timestamps(self,format=None):
        """Get the timestamps of all of the events in the array

        Args:
            format (str, optional): The time format that you want to export the events to. Defaults to None.

        Returns:
            timestamps (np.array): The timestamps of all of the events in the predefined format
        """
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
        """Load events from a file from the FERMI data

        Args:
            filename (str): The filename of the file to load
            append (bool, optional): If True then appends them to the current array, if False it returns an additional array. Defaults to True.

        Returns:
            events (array) : If append is False then return the array of events
        """
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
        """Load events from a file from the lightnings data

        Args:
            filename (str): The filename of the file to load
            append (bool, optional): If True then appends them to the current array, if False it returns an additional array. Defaults to True.

        Returns:
            events (array) : If append is False then return the array of events
        """
        # Load the lightning data
        data    = pd.read_csv(filename).to_numpy()
        lights  = []                                   # List to store the ligtnigs

        # For all the lightings in the loaded dataset
        for datum in data:
            # Create a TGF and append it to the array
            lights.append(event(timestamp   = float(datum[0]) * 1e-9,
                                longitude   = in_range(float(datum[2])), 
                                latitude    = float(datum[1]),
                                detector_id = 'Blitz',
                                event_id    = 'li',
                                mission     = 'Blitzurtong',
                                time_format = 'unix',
                                event_type  = 'Lightning'))
        
        # If you want to append the data to the original array do so here
        if append: self.events += lights

        # Otherwise return them as a different tgf_array
        else: return array(lights)

    # Generate an array of cubesat locations
    def location_from_file(self,filename:str,append:bool=True):
        """Load events from a file from the cubesat locations

        Args:
            filename (str): The filename of the file to load
            append (bool, optional): If True then appends them to the current array, if False it returns an additional array. Defaults to True.

        Returns:
            events (array) : If append is False then return the array of events
        """
        # Load the location data
        data    = pd.read_csv(filename).to_numpy()
        locs    = []

        # For all the possible rows
        for datum in data:
            locs.append(event(timestamp     = Time.strptime(datum[0],'%d/%m/%Y %H:%M:%S.%f').to_value(format='unix'),
                                longitude   = in_range(float(datum[2])), 
                                latitude    = float(datum[1]),
                                detector_id = 'NA',
                                event_id    = 'id',
                                mission     = 'NanoAvionics',
                                time_format = 'unix',
                                event_type  = 'cubesat-location'))

        # If you want to append the data to the original array do so here
        if append: self.events += locs

        # Otherwise return them as a different location_array
        else: return array(locs)
    
    def from_file(self,filename:str,event_type:str=event_types[0],append:bool=True):
        # Choose the appropriate function to load the data
        if   event_type == array.event_types[0]:
            return self.fermi_from_file(filename=filename, append=append)
        
        elif event_type == array.event_types[2]:
            return self.lightning_from_file(filename=filename, append=append)
        
        elif event_type == array.event_types[3]:
            return self.location_from_file(filename=filename,append=append)
