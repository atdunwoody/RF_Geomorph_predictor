import pygrib
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def generate_date_range(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return [start + timedelta(days=x) for x in range(0, (end-start).days + 1)]

def construct_urls(base_url, date_range):
    return [f"{base_url}{date.strftime('%Y')}/{date.strftime('%m')}/{date.strftime('%d')}/mrms/ncep/PrecipFlag/" for date in date_range]

def download_file(file_url, download_dir):
    try:
        filename = os.path.join(download_dir, os.path.basename(file_url))
        # Check if file already exists
        if os.path.exists(filename):
            print(f"File {filename} already exists, skipping download.")
            return False

        file_response = requests.get(file_url)
        file_response.raise_for_status()
        
        with open(filename, 'wb') as file:
            file.write(file_response.content)
        return True
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except Exception as e:
        print(f"Error: {e}")
    return False

def download_files_from_url(url, download_dir):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        files = [url + link.get('href') for link in soup.find_all('a') if link.get('href') and link.get('href').endswith('.grib2.gz')]
        date = "-".join(url.split('/')[-7:-4])
        
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda file_url: download_file(file_url, download_dir), files), total=len(files), desc=f"Downloading hourly data for {date}", leave=False))
            
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except Exception as e:
        print(f"Error: {e}")

def main(start_date, end_date, download_dir='downloaded_files'):
    base_url = 'https://mtarchive.geol.iastate.edu/'
    os.makedirs(download_dir, exist_ok=True)
    
    date_range = generate_date_range(start_date, end_date)
    urls = construct_urls(base_url, date_range)
    
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda url: download_files_from_url(url, download_dir), urls), total=len(urls), desc=f"Downloading data for date range {start_date} to {end_date}"))

def read_grib2_file(filepath):
    with pygrib.open(filepath) as grbs:
        for grb in grbs:
            print(grb)
            data = grb.values

if __name__ == '__main__':
    years = [2021, 2022, 2023]
    for year in years:
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        download_dir = os.path.join(r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA", str(year)) 
        print(f"Downloading data for date range {start_date} to {end_date}")
        print(f"Files will be saved to {download_dir}")
        main(start_date, end_date, download_dir)
