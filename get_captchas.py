import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

parser = argparse.ArgumentParser(description="Get Captchas from the Server")
parser.add_argument('--short-username', help='Short username of the user', type=str)
parser.add_argument('--output-folder-name', help='Folder where the downloaded files should be saved', type=str)
parser.add_argument('--file-list-name', help='File where all the captcha filenames to be downloaded are present', type=str)

args = parser.parse_args()

if args.short_username is None or args.output_folder_name is None or args.file_list_name is None:
    print("Please specify all required arguments (short_username, output folder name and filelist name)")
    exit(1)

short_username = args.short_username
output_folder_name = args.output_folder_name
list_of_files = args.file_list_name

files = []
with open(list_of_files) as f:
    files = f.read().splitlines()
file_list = files

base_url = f'https://cs7ns1.scss.tcd.ie?shortname={short_username}&myfilename='
output_dir = f'{output_folder_name}/captchas/'  
total_files = 4000  
retry_attempts = 5  
max_workers = 10  
interval_between_batches = 5 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def download_file(file_name):
    file_url = f"{base_url}{file_name}"
    file_path = os.path.join(output_dir, file_name)

    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()  
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Successfully downloaded {file_name}")
        return file_name, True  
    except Exception as e:
        print(f"Error downloading {file_name} from {file_url}: {e}")
        return file_name, False 

def download_files_concurrently(file_list, max_workers):
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, file_name): file_name for file_name in file_list}
        
        for future in as_completed(futures):
            file_name, success = future.result()
            if not success:
                failed_files.append(file_name)
            elif success and file_name in failed_files:
                failed_files.remove(file_name)
    
    return failed_files

def fallback_download(failed_files, retry_attempts, max_workers):
    for attempt in range(retry_attempts):
        if not failed_files:
            print("All files downloaded successfully.")
            break
        
        print(f"Retry attempt {attempt + 1}/{retry_attempts} for failed files.")
        failed_files = download_files_concurrently(failed_files, max_workers)
        
        if not failed_files:
            print("Successfully downloaded all files after retry attempts.")
            break


failed_files = download_files_concurrently(file_list, max_workers)

if interval_between_batches > 0:
    time.sleep(interval_between_batches)

while len(failed_files) > 0:
    fallback_download(failed_files, retry_attempts, max_workers)
else:
    print("All files downloaded successfully")
