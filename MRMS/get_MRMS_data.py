import pygrib
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta
from tqdm import tqdm  # Import tqdm

def generate_date_range(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = [start + timedelta(days=x) for x in range(0, (end-start).days + 1)]
    return date_range

def construct_urls(base_url, date_range):
    urls = []
    for date in date_range:
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        url = f"{base_url}{year}/{month}/{day}/mrms/ncep/MultiSensor_QPE_01H_Pass2/"
        #url = f"{base_url}{year}/{month}/{day}/mrms/ncep/GaugeCorr_QPE_01H/"
        urls.append(url)
    return urls

def download_files_from_url(url, download_dir):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all .grib2.gz links and store in list
        files = [link.get('href') for link in soup.find_all('a') if link.get('href') and link.get('href').endswith('.grib2.gz')]
        date = url.split('/')[-7:-4]
        date =f"{date[0]}-{date[1]}-{date[2]}"
        for file_name in tqdm(files, desc=f"Downloading hourly data for {date}", leave=False):
            file_url = url + file_name
            file_response = requests.get(file_url)
            file_response.raise_for_status()
            
            filename = os.path.join(download_dir, file_name.split('/')[-1])
            with open(filename, 'wb') as file:
                file.write(file_response.content)
            #print(f"Downloaded {filename}")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except Exception as e:
        print(f"Error: {e}")

def main(start_date, end_date, download_dir='downloaded_files'):
    base_url = 'https://mtarchive.geol.iastate.edu/'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    date_range = generate_date_range(start_date, end_date)
    urls = construct_urls(base_url, date_range)
    
    for url in tqdm(urls, desc=f"Downloading hourly data for date range {start_date} to {end_date}"):
        download_files_from_url(url, download_dir)

def read_grib2_file(filepath):
    grbs = pygrib.open(filepath)
    for grb in grbs:
        print(grb)
        data = grb.values
    grbs.close()

if __name__ == '__main__':
    start_date = '2020-10-13'
    end_date = '2020-12-31'
    download_dir = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Precipitation\MRMS Data\2020_Multi_QPE"
    main(start_date, end_date, download_dir)
