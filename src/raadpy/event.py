#############################
#     RAAD event Class      #
#############################

from .core import *

# Event Class
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