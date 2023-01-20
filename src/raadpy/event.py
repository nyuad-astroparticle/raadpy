#############################
#     RAAD event Class      #
#############################

from .core import *

# Event Class
class event:
    # Constructor ###############################################################
    def __init__(self,timestamp,latitude:float,longitude:float,detector_id:str,event_id:str='',mission:str='',time_format:str='mjd',event_type:str='TGF', property:float=0):
        """Event: A point with longitude and latitude over the earth with a timestamp

        Args:
            timestamp (_type_): Either string or Astropy.Time object with the timestamp
            latitude (float): Latitude
            longitude (float): Longitude
            detector_id (str): The id of the detector, can be anything
            event_id (str, optional): ID of event, can be anything. Defaults to ''.
            mission (str, optional): The name of the mission, e.g. Fermi. Defaults to ''.
            time_format (str, optional): The time format of the timestamp for conversion. Defaults to 'mjd'.
            event_type (str, optional): What is the event?. Defaults to 'TGF'.
        """
        # Set up the variables
        self.timestamp      = Time(timestamp, format=time_format)
        self.latitude       = latitude
        self.longitude      = longitude
        self.detector_id    = detector_id
        self.event_id       = event_id
        self.mission        = mission
        self.event_type     = event_type
        self.property       = property

    
    # SOME FUNCTIONS FOR THE TGF CLASS ##########################################

    # Return a string for that TGF
    def __str__(self):
        str = ''' %s: %s | Mission: %s
        Timestamp (ISO): %s
        Lat: %9.4f \t Long: %9.4f
        Detector_id: %s
        Property: %f
        '''%(self.event_type,self.event_id,self.mission,self.get_iso(),self.latitude,self.longitude,self.detector_id,self.property)

        return str

    # Print a TGF
    def print(self):
       print(self.__str__())

    # Get any time format that astropy has to offer
    def get_timestamp(self,time_format:str='mjd',data_type:str='long'):
        """Return the timestamp in any format

        Args:
            time_format (str, optional): What format do you want. Defaults to 'mjd'.
            data_type (str, optional): If it is a numerical format what is the precision?. Defaults to 'long'.

        Returns:
            str/data_type: The converted timestamp
        """
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