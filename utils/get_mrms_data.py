# %%
from datetime import datetime, time
import os
import urllib.request
from urllib.request import HTTPError
from datetime import timedelta

# source for code: https://github.com/HydrologicEngineeringCenter/data-retrieval-scripts/blob/master/retrieve_qpe_gagecorr_01h.py
# 
# https://www.hec.usace.army.mil/confluence/hmsdocs/hmsguides/working-with-gridded-boundary-condition-data/downloading-multi-radar-multi-sensor-mrms-precipitation-data
# 

# 
# Unzip files after download

# change dates and sample interval

def progress_bar(count, total_size):
    percent = int(count / total_size *100)
    print(f"Downloading... {percent}%", end="\r", flush=True)

start = datetime(2023, 6, 3, 14, 0)
end = datetime(2023, 6, 30, 23, 58)
minute = timedelta(minutes=2)
#minute = timedelta(hours=1)
# range of months to download data for
mo1 = 1
mo2 = 12

# indicate which product to download (rate,RQI,QPE_multi,QPE_radar)
product = 'QPE_multi'

# tell program where to download files ==> /data/raw
# Create a path to the code file
codeDir = os.path.dirname(os.path.abspath(os.getcwd()))
# Create a path to the data folder
# destination = os.path.join(codeDir,"data","processed")
destination = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\MRMS Data"
# Change to data folder
os.chdir(destination)

missing_dates = []
fallback_to_radaronly = True #Enables a post-processing step that will go through the list of missing dates for gage-corrected
############################# and tries to go get the radar-only values if they exist.

date = start
total_days = (end-start).days
print(f"Downloading data for {total_days} days")
while date <= end:
    progress_bar(((date.year + date.month + date.day+date.hour)-(start.year + start.month + start.day + start.hour)), ((end.year + end.month + end.day + end.hour)-(start.year + start.month + start.day+start.hour)))
    if date.month>=mo1 and date.month<=mo2:
        if product == 'rate':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/PrecipRate/PrecipRate_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(
            date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        
        elif product == 'RQI':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/RadarQualityIndex/RadarQualityIndex_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        elif product == 'QPE_multi':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/MultiSensor_QPE_01H_Pass2/MultiSensor_QPE_01H_Pass2_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        elif product == 'QPE_multi':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/RadarOnly_QPE_01H/RadarOnly_QPE_01H_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        
        filename = url.split("/")[-1]
        try:
            fetched_request = urllib.request.urlopen(url)
        except HTTPError as e:
            missing_dates.append(date)
        else:
            with open(destination + os.sep + filename, 'wb') as f:
                f.write(fetched_request.read())
        finally:
            date += minute
    else:
        date += minute

if fallback_to_radaronly:
    radar_also_missing = []
    for date in missing_dates:
        if product == 'rate':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/PrecipRate/PrecipRate_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(
            date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        
        elif product == 'RQI':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/RadarQualityIndex/RadarQualityIndex_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        elif product == 'QPE_multi':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/MultiSensor_QPE_01H_Pass2/MultiSensor_QPE_01H_Pass2_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        elif product == 'QPE_multi':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/RadarOnly_QPE_01H/RadarOnly_QPE_01H_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        
        
        filename = url.split("/")[-1]
        try:
            fetched_request = urllib.request.urlopen(url)
        except HTTPError as e:
            radar_also_missing.append(date)
        else:
            with open(destination + os.sep + filename, 'wb') as f:
                f.write(fetched_request.read())

print(radar_also_missing)






# %%
