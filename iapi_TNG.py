"""
Contains useful funcitons for fetching data with the IllustrisTNG API
For the illustrisTNG workshop at STScI Symposium April 2024
Modified by Bryanne McDonough from materials provided by the TNG team
"""


import requests
import numpy as np
import h5py
import os.path

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key" : "baa709f61f16a785a1864e5c4526add9"}

def get(path, params=None, fName='temp'): # gets data from url, saves to file
    """
    Routine to pull data from online
    Credit to TNG team
    """
    
    # make HTTP GET request to path
    if (len(headers['api-key'])!=32):
        print("Check your api key")
    r = requests.get(path, params=params, headers=headers)
    
    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    dataFile=fName+'.hdf5'
    # Saves to file, currently disabled
    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(dataFile, 'wb') as f:
            f.write(r.content)
        return dataFile # return the filename string

    return r


def getsub(snapnum,subid):
    #Pull fields associated with a given subhalo at a given snapshot
    url= 'https://www.tng-project.org/api/TNG100-1/snapshots/'+str(snapnum)+'/subhalos/'+str(subid)+'/'
    sub=iapi.get(url)
    return(sub)

def gettree(snapnum,subid):
    snapnum=str(snapnum)
    fName = 'trees/sublink_mpb_'+str(subid)
    if os.path.exists(fName+'.hdf5'):
        return(fName+'.hdf5')
    url='https://www.tng-project.org/api/TNG100-1/snapshots/'+snapnum+'/subhalos/'+str(subid)+'/sublink/mpb.hdf5'
    tree=get(url,fName=fName)
    return(tree)

def getredshift(snapnum):
    
    r=get(baseUrl)
    names = [sim['name'] for sim in r['simulations']]
    i = names.index('TNG100-1')
    sim = get( r['simulations'][i]['url'] )
    
    snaps = get( sim['snapshots'] )
    
    try: 
        zs=[snaps[j]['redshift'] for j in snapnum]
        return(zs)
    except: return(snaps[snapnum]['redshift'])







def getSubhaloField(field, simulation='TNG100-1', snapshot=99,
                    fileName='tempCat', rewriteFile=0):
    """
    Credit to TNG team
    Data from one field for all subhalos in a given snapshot      
    
    These two commands are near identical, so I'm going to detail them both here. 
    They have the same input and output, except one deals with halos (roughly 
    speaking 'groups/ clusters') and the other with the subhalos in those halos 
    (the 'galaxies' in those 'groups'). See Intro to the Data (or Naming 
    Conventions) for more on the data structure/ naming conventions used.
    
    
    Parameters
    ----------
    field : str
        Name of the one field to be returned. The fields can be found in 
        section 2. of this page
        http://www.illustris-project.org/data/docs/specifications/

    simulation : str
        Which simulation to pull data from

    snapshot : int
        Which snapshot (moment in time) to pull data from

    The following two parameters are discussed in more detail here!

    fileName : str
        Default is 'tempGal.hdf5'. Filename for where to store or load the data 

    rewriteFile : int
        [0 or 1] If this is equal to 0 then the program will try and pull data 
        from the file specified by fileName rather than re-downloading. This can 
        save time, especially for galaxies which are large or you will work on 
        frequently, but you will only be able to access fields you originally 
        requested
        
        
    Returns
    -------
    data : array
        A numpy array containing the data for a specific field for all halos/subhalos

        
    Examples
    --------
    Let's pull out the velocity dispersion of stars in every subhalo and their 
    DM mass, and then restrict ourselves to only looking at the primary subhalo 
    in each halo (i.e. the most massive galaxy in each group).

    The velocity dispersion (N_sub values)
    
        >>> galaxyVelDisp=iApi.getSubhaloField('SubhaloVelDisp')

    The mass of each different particle type in a galaxy (N_sub x 6 values, 
    see getGalaxyData for more info on particle types)
    
        >>> galaxyMassType=iApi.getSubhaloField('SubhaloMassType') 

    The subhalo number of the primary subhalo in each halo (N_halo values)
        
        >>> primarySubhalos=iApi.getHaloField('GroupFirstSub') 

    Velocity dispersion of primary subhalos
    
        >>> velDisp=galaxyVelDisp[primarySubhalos]

    Total dark matter mass of primary subhalos 
    
        >>> mDM=galaxyMassType[primarySubhalos,1] 
    
    """

    dataFile=fileName+'.hdf5'
    
    if not os.path.exists(dataFile) or rewriteFile==1:
        url='http://www.tng-project.org/api/'+simulation+'/files/groupcat-'+str(snapshot)+'/?Subhalo='+field
        dataFile=get(url,fName=fileName)
        
    with h5py.File(dataFile,'r') as f:
        data=np.array(f['Subhalo'][field])

    return data
    
  
def getHaloField(field, simulation='TNG100-1', snapshot=99,
                 fileName='tempCat', rewriteFile=0):
    """
    Credit to TNG team
    Data from one field for all halos/subhalos in a given snapshot      
    
    These two commands are near identical, so I'm going to detail them both here. 
    They have the same input and output, except one deals with halos (roughly 
    speaking 'groups/ clusters') and the other with the subhalos in those halos 
    (the 'galaxies' in those 'groups'). See Intro to the Data (or Naming 
    Conventions) for more on the data structure/ naming conventions used.
    
    
    Parameters
    ----------
    field : str
        Name of the one field to be returned. The fields can be found in 
        section 2. of this page
        http://www.illustris-project.org/data/docs/specifications/

    simulation : str
        Which simulation to pull data from

    snapshot : int
        Which snapshot (moment in time) to pull data from

    The following two parameters are discussed in more detail here!

    fileName : str
        Default is 'tempGal.hdf5'. Filename for where to store or load the data 

    rewriteFile : int
        [0 or 1] If this is equal to 0 then the program will try and pull data 
        from the file specified by fileName rather than re-downloading. This can 
        save time, especially for galaxies which are large or you will work on 
        frequently, but you will only be able to access fields you originally 
        requested
        
        
    Returns
    -------
    data : array
        A numpy array containing the data for a specific field for all halos/subhalos

        
    Examples
    --------
    Let's pull out the velocity dispersion of stars in every subhalo and their 
    DM mass, and then restrict ourselves to only looking at the primary subhalo 
    in each halo (i.e. the most massive galaxy in each group).

    The velocity dispersion (N_sub values)
    
        >>> galaxyVelDisp=iApi.getSubhaloField('SubhaloVelDisp')

    The mass of each different particle type in a galaxy (N_sub x 6 values, 
    see getGalaxyData for more info on particle types)
    
        >>> galaxyMassType=iApi.getSubhaloField('SubhaloMassType') 

    The subhalo number of the primary subhalo in each halo (N_halo values)
        
        >>> primarySubhalos=iApi.getHaloField('GroupFirstSub') 

    Velocity dispersion of primary subhalos
    
        >>> velDisp=galaxyVelDisp[primarySubhalos]

    Total dark matter mass of primary subhalos 
    
        >>> mDM=galaxyMassType[primarySubhalos,1] 
    
    """
    dataFile=fileName+'.hdf5'
    
    if not os.path.exists(dataFile) or rewriteFile==1:
        print('i did it')
        url='http://www.tng-project.org/api/'+simulation+'/files/groupcat-'+str(snapshot)+'/?Group='+field
        dataFile=get(url,fName=fileName)

        
    with h5py.File(dataFile,'r') as f:
        data=np.array(f['Group'][field])

    return data

    
def getSubcutout(subID, parttype, params, sim='TNG100-1', snapnum='99', fName='temp'):
    """
    Obtain particle level information (params) for all particles of parttype bound to the subhalo identified by subID
    subID (int): index into subhalo catalog
    parttype (str): can be 'gas', 'dm', 'stars', 'bhs', or 'tracers'
    params (str): parameters in the PartType fields to pull, formatted as: 'param1, param2, param3, ..., paramN'
    sim (str): simulation to pull the cutout from
    snapnum (int or str): snap number to pull cutout from
    fName (str): file name to save cutout to
    """
    
    if fName!='temp' and os.path.exists(fName):
        return(fName)
    
    snapnum=str(snapnum)
    part_param = {parttype : params}
    
    sub_url = "http://www.tng-project.org/api/"+sim+"/snapshots/"+snapnum+"/subhalos/"+str(subID)+"/"
    sub=get(sub_url)
    print(sub_url)
    cutouturl=sub['cutouts']['subhalo']
    
    
    cutout = get(cutouturl,params=part_param, fName=fName)
    
    return(cutout)

