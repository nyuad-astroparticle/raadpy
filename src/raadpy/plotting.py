#############################
#  RAAD Plotting functions  #
#############################

from .core import *
from .rparray import array
from .event import *
from .functionality import *


# Visualize 2 sets of points on a map
def map(list1,list2:array=None,name1='',name2='',size:int=500,long=-90,lat=30):
    """Plot up to two lists of raadpy events on an interactive globe

    Args:
        list1 (_type_): Either a single event object or a raadpy array of events
        list2 (array, optional): A raadpy array of events. If None it will only plot the first argument. Defaults to None.
        name1 (str, optional): Label for the first list. Defaults to ''.
        name2 (str, optional): Label for the second list. Defaults to ''.
        size (int, optional): Figure size in pixels. Defaults to 500.
        long (int, optional): Longitude of the center of the map. Defaults to -90.
        lat (int, optional): Latitude of the center of them. Defaults to 30.

    Returns:
        fig (plotly): Plotly interactive figure
    """
    # If it is a single point, convert it into an array
    if type(list1)   == list:        list1 = array(list1)
    elif type(list1) == event:       list1 = array([list1])
    elif type(list1) != array: raise Exception("type %s is not an event object nor a list. Please enter a TGF object"%type(list1))

    # Create the map
    fig = go.Figure(data = go.Scattergeo(
        name=name1,
        lon=list1.get_coords().T[0],
        lat=list1.get_coords().T[1],
        text=list1.get_timestamps(format='iso'),
        mode = 'markers',
        marker=dict(
            size=4,
            color='blue',
            symbol = 'circle-dot'
        ),
    ))
    
    # if there are lightnings create the lighning map and add it to the TGF map
    if list2 is not None:
        fig.add_trace(go.Scattergeo(
            name=name2,
            lon=list2.get_coords().T[0],
            lat=list2.get_coords().T[1],
            text=list2.get_timestamps(format='iso'),
            mode = 'markers',
            marker=dict(
                size=4,
                color='red',
                symbol = 'circle-dot'
            ),
        ))
        

    fig.update_geos(projection_type="orthographic",projection_rotation=dict(lon=long, lat=lat))
    fig.update_layout(height=size,width=size, margin={"r":0,"t":0,"l":0,"b":0})

    return fig

# Plot the dictionary data obtained from a buffer:
def plot_buffer(data,title='Plots of Buffer data',UNITS=None):
    """Plot buffer automatically

    Args:
        data (_type_): Buffer that we want to plot
        title (str, optional): Title of the plot. Defaults to 'Plots of Buffer data'.
        UNITS (_type_,optional): List of strings to be used as units for the plots.

    Returns:
        fig,ax: Matplotlib figure and axis
    """
    # Get the keys and event numbers
    keys    = list(data.keys())
    events  = range(len(data[keys[0]]))
    colors  = cm.get_cmap('Dark2').colors

    # Create a figure
    fig, axes = plt.subplots((len(keys)+1)//2,2,sharex=True,figsize=(14,4*len(keys)//2),dpi=100)
    fig.subplots_adjust(top=0.95)
    fig.suptitle(title,fontsize=18)
    axes = axes.flatten()

    # Handle units
    if UNITS is None: UNITS = ['']*len(keys)
    assert(len(UNITS) == len(keys))

    # Plot each of the data points
    for i,key,ax,unit in zip(range(len(axes)),keys,axes,UNITS):
        ax.plot(events,data[key],c=colors[i%len(colors)],lw=0.7)
        ax.set_title(key.title()+' '+unit)

        # Adding x tick labels
        if i != len(axes) -1:
            ax.xaxis.set_tick_params(which='both',labelbottom=True)

        # Customize the plot style
        ax.tick_params(axis='both',which='both',direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(axis='both', which='major', lw=0.25)
        ax.grid(axis='both', which='minor', lw=0.2, ls=':')
        

    return fig,axes

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
    for channel in np.unique(data['channel']):
        idx         = np.where(data['channel'] == channel)[0]
        channels.append(dict(zip(struct.keys(),[arr[idx] for arr in data.values()])))
    
    return channels

# Plot histograms of the energies
def plot_hists(data,struct=NONVETO_STRUCT,bins=600,RANGE=None):
    """Plot histograms of the charge (ADC counts) given decoded data from one of the dictionaries

    Args:
        data (dict): The decoded data
        struct (dict, optional): The structure dictionaryof the data in bits. Defaults to NONVETO_STRUCT.
        bins (int, optional): Number of bins for the histograms. Defaults to 600.
        RANGE (tuple, optional): Tuple of floats for the x-axis range. Defaults to None.

    Returns:
        fig,ax: matplotlib figure descriptions
    """
    # Get the splitted channels
    channels = split_channels(data,struct)

    # Create a figure
    fig,ax  = plt.subplots(len(channels),1,figsize=(14,4*len(channels)),dpi=100,sharex=True)
    ax      = ax.flatten()
    colors  = cm.get_cmap('Dark2').colors

    # Get the maximum energy
    ADC_max = max([max(channel['adc_counts']) for channel in channels])


    # Plot the histogram of each channel
    for i,channel in enumerate(channels):
        ax[i].hist(channel['adc_counts'],bins=int(bins/ADC_max*max(channel['adc_counts'])),range=RANGE,color=colors[i%len(channels)])

        ax[i].set_title('Energy of Channel: %d'%i)
        ax[i].set_yscale('log')
        ax[i].tick_params(axis='both',which='both',direction='in',top=True,right=True)
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].grid(axis='both', which='major', lw=0.25)
        ax[i].grid(axis='both', which='minor', lw=0.2, ls=':')

    return fig,ax

# Plots the timestamps of the measurements by channel
def plot_timestamps(data,struct=NONVETO_STRUCT,RANGE=None):
    """Plot timestamps of the measurements by channel number

    Args:
        data (_type_): Decoded dictionary of the data buffers
        struct (_type_, optional): The stucture dictionary of the data in bits. Defaults to NONVETO_STRUCT.
        RANGE (_type_, optional): Tuple of integets to define the index range of the plots . Defaults to None.

    Returns:
        fig,ax: matplotlib figure descriptions
    """
    # Get the splitted channels
    channels = split_channels(data,struct)

    # Create a figure
    fig,ax  = plt.subplots(len(channels),1,figsize=(14,4*len(channels)),dpi=100,sharex=True)
    ax      = ax.flatten()
    colors  = cm.get_cmap('Dark2').colors

    # Plot the histogram of each channel
    for i,channel in enumerate(channels):
        length   = len(channel['stimestamp'])
        if RANGE is None: _RANGE = (0,length)
        else: _RANGE = RANGE

        ax[i].plot   (range(*_RANGE),channel['stimestamp'][_RANGE[0]:_RANGE[1]],c=to_hex(colors[i%len(channels)]),lw=0.4)
        ax[i].scatter(range(*_RANGE),channel['stimestamp'][_RANGE[0]:_RANGE[1]],c=to_hex(colors[i%len(channels)]),marker='o',s=2)

        ax[i].set_title('Timestamp of Channel: %d'%i)
        ax[i].tick_params(axis='both',which='both',direction='in',top=True,right=True)
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
        ax[i].grid(axis='both', which='major', lw=0.25)
        ax[i].grid(axis='both', which='minor', lw=0.2, ls=':')

    return fig,ax

# Plot the timestamp of a data dictionary
def plot_timestamp(data,RANGE=None):
    """Plot the timstamp of a buffer

    Args:
        data (_type_): Decoded dictionary of the data buffers
        RANGE (_type_, optional): Index range of the plot. Defaults to None.

    Returns:
        fig,ax: matplotlib figure descriptions
    """
    # Create a figure
    fig     = plt.figure(figsize=(15,4),dpi=100)
    ax      = fig.add_subplot(111)

    length   = len(data['stimestamp'])
    if RANGE is None: RANGE = (0,length)

    ax.plot   (range(*RANGE),data['stimestamp'][RANGE[0]:RANGE[1]],c='k',lw=0.4)
    ax.scatter(range(*RANGE),data['stimestamp'][RANGE[0]:RANGE[1]],c='k',marker='o',s=2)

    ax.set_title('Timestamps vs event number')
    ax.tick_params(axis='both',which='both',direction='in',top=True,right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis='both', which='major', lw=0.25)
    ax.grid(axis='both', which='minor', lw=0.2, ls=':')

    return fig,ax