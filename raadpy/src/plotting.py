#############################
#  RAAD Plotting functions  #
#############################

from core import *
from __array import array
from event import *
from functionality import *

# Visualize a set of TGFs on a map
def map(tgfs,lightnings:array=None):
    gv.extension('bokeh', 'matplotlib');
    
    # If it is a single point, convert it into an array
    if type(tgfs)   == list:        tgfs = array(tgfs)
    elif type(tgfs) == event:       tgfs = array([tgfs])
    elif type(tgfs) != array: raise Exception("type %s is not an event object nor a list. Please enter a TGF object"%type(tgfs))

    # Convert the points to GeoViews points
    points = gv.Points([tgfs.get_coords()])

    # Create the tgf map
    tgf_map = (gv.tile_sources.OSM * points).opts(gv.opts.Points(
                global_extent=True, 
                width=1300, 
                height=900, 
                size=7,
                color='Blue',
                marker='+'))
    
    # if there are lightnings create the lighning map and add it to the TGF map
    if lightnings is not None:
        points_lightning = gv.Points([lightnings.get_coords()])
        
        lightning_map = (gv.tile_sources.OSM * points_lightning).opts(gv.opts.Points(
                global_extent=True, 
                width=1300, 
                height=900, 
                size=7,
                color='red',
                marker='+'))

        tgf_map *= lightning_map

    return tgf_map

# Plot the dictionary data obtained from a buffer:
def plot_buffer(data,title='Plots of Buffer data'):
    # Get the keys and event numbers
    keys    = list(data.keys())
    events  = range(len(data[keys[0]]))
    colors  = cm.get_cmap('Dark2').colors

    # Create a figure
    fig, axes = plt.subplots((len(keys)+1)//2,2,sharex=True,figsize=(20,4*len(keys)//2),dpi=200)
    fig.subplots_adjust(top=0.95)
    fig.suptitle(title,fontsize=18)
    axes = axes.flatten()

    # Plot each of the data points
    for i,key,ax in zip(range(len(axes)),keys,axes):
        ax.plot(events,data[key],c=colors[i%len(colors)],lw=0.7)
        ax.set_title(key.title())

        # Customize the plot style
        ax.tick_params(axis='both',which='both',direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(axis='both', which='major', lw=0.25)
        ax.grid(axis='both', which='minor', lw=0.2, ls=':')
        

    return fig,axes

# Split the dataset in channels
def split_channels(data,struct=NONVETO_STRUCT):
    # Split the data based on their channels
    channels    = []
    for channel in np.unique(data['channel']):
        idx         = np.where(data['channel'] == channel)[0]
        channels.append(dict(zip(struct.keys(),[arr[idx] for arr in data.values()])))
    
    return channels

# Plot histograms of the energies
def plot_hists(data,struct=NONVETO_STRUCT,bins=600,RANGE=None):
    # Get the splitted channels
    channels = split_channels(data,struct)

    # Create a figure
    fig,ax  = plt.subplots(len(channels),1,figsize=(15,4*len(channels)),dpi=200,sharex=True)
    ax      = ax.flatten()
    colors  = cm.get_cmap('Dark2').colors

    # Plot the histogram of each channel
    for i,channel in enumerate(channels):
        ax[i].hist(channel['adc_counts'],bins=bins,range=RANGE,color=colors[i%len(channels)])

        ax[i].set_title('Energy of Channel: %d'%i)
        ax[i].set_yscale('log')
        ax[i].tick_params(axis='both',which='both',direction='in',top=True,right=True)
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].grid(axis='both', which='major', lw=0.25)
        ax[i].grid(axis='both', which='minor', lw=0.2, ls=':')

    return fig,ax

# Plots the timestamps of the measurements by channel
def plot_timestamps(data,struct=NONVETO_STRUCT,RANGE=None):
    # Get the splitted channels
    channels = split_channels(data,struct)

    # Create a figure
    fig,ax  = plt.subplots(len(channels),1,figsize=(15,4*len(channels)),dpi=200,sharex=True)
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
