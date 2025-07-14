import requests
import zipfile
import os
dropbox_link = "https://www.dropbox.com/scl/fi/t44o5k8c0gaazyngprorf/data.zip?rlkey=iofmmyn2sawfefusesvod3nr3&st=m1qj319q&dl=1"
download_path = "./data.zip"
targeted_dir = "./data/"

def download_data(download_link, download_path):
    """
    Description:
    Downloads a file from the specified URL and saves it to the given directory.
    ====== Parameters ======
    @param download_link: The URL to download the file from.
    @param download_dir: The path to which to save the data.
    ====== Returns ======
    @return: None. Prints status messages.
    """
    print(f"Downloading from Dropbox...\nURL: {download_link}")

    try:
        response = requests.get(download_link, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        with open(download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Download complete. File saved to: {download_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")

def extract_zip_file(file_path, targeted_dir, exclude_files = ['__MACOSX/']):
    """
    Description: Extracts the contents of a ZIP file to the specified directory.
    ====== Parameters ======
    @param file_path: The path to the ZIP file.
    @param targeted_dir: The directory where the contents should be extracted.
    ====== returns =====
    @return: None if file doesn't exist or an error occurs. Otherwise, prints status messages.
    """
    if (not os.path.exists(file_path)):
        print("Error: File doesn't exist")
        return None
    try:
        exclude_files = exclude_files or []
        with zipfile.ZipFile(file_path, "r") as zip:
            for member in zip.infolist():
                if any(member.filename.startswith(prefix) for prefix in exclude_files):
                    continue
                zip.extract(member, targeted_dir)
    except zipfile.BadZipFile:
        print(f"Error: File {file_path} is not a valid zip file or is corrupted.")
    except zipfile.LargeZipFile:
        print(f"Error: ZIP file requires ZIP64 support but it's not enabled.")
