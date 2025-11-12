import os
import gdown
import argparse
import urllib.request
import zipfile
import requests
import shutil

def download_datasets():
    """
    Download datasets ZIP file from Google Drive and extract it.
    Removes existing datasets folder if it exists.
    """
    
    # Google Drive file ID extracted from your share link
    file_id = "1i4qVwj90w_uhBIwCGoJuI6gsXAcMtjgN"
    
    # Define datasets directory
    datasets_dir = "./data/"
    
    # Remove existing datasets folder if it exists
    if os.path.exists(datasets_dir):
        print(f"🗑️  Removing existing datasets folder: {datasets_dir}")
        shutil.rmtree(datasets_dir)
        print("✅ Existing datasets folder removed")
    
    # Create fresh datasets directory
    os.makedirs(datasets_dir, exist_ok=True)
    print(f"📁 Created fresh datasets directory: {datasets_dir}")
    
    # Define the ZIP file path
    zip_path = os.path.join(datasets_dir, "datasets.zip")
    
    print(f"Downloading datasets ZIP file to: {zip_path}")
    print("This may take a while depending on the file size...")
    
    # Try multiple download methods
    download_success = False
    
    # Method 1: Standard gdown download
    try:
        print("🔄 Attempting standard gdown download...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False)
        download_success = True
        print(f"✅ ZIP file downloaded successfully using gdown")
        
    except Exception as e:
        print(f"❌ Standard gdown failed: {str(e)}")
        
        # Method 2: Try with fuzzy download (handles permission issues better)
        try:
            print("🔄 Attempting fuzzy download...")
            gdown.download(f"https://drive.google.com/file/d/{file_id}/view?usp=sharing", 
                          zip_path, quiet=False, fuzzy=True)
            download_success = True
            print(f"✅ ZIP file downloaded successfully using fuzzy method")
            
        except Exception as e2:
            print(f"❌ Fuzzy download failed: {str(e2)}")
            
            # Method 3: Try direct requests download with virus scan bypass
            try:
                print("🔄 Attempting direct requests download with virus scan bypass...")
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                
                session = requests.Session()
                response = session.get(download_url, stream=True)
                
                # Handle Google Drive virus scan warning for large files
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        params = {'id': file_id, 'confirm': value}
                        response = session.get(download_url, params=params, stream=True)
                        break
                
                # Also try with confirm=t for large files
                if 'virus scan' in response.text.lower() or response.status_code != 200:
                    print("🔄 Detected virus scan warning, bypassing...")
                    params = {'id': file_id, 'confirm': 't'}
                    response = session.get(download_url, params=params, stream=True)
                
                if response.status_code == 200:
                    print("📥 Downloading large file (bypassing virus scan)...")
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    percent = (downloaded / total_size) * 100
                                    print(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)", end="")
                    
                    print()  # New line after progress
                    download_success = True
                    print(f"✅ ZIP file downloaded successfully using requests (virus scan bypassed)")
                else:
                    raise Exception(f"HTTP {response.status_code}")
                    
            except Exception as e3:
                print(f"❌ Direct download failed: {str(e3)}")

    # If download was successful, extract the file
    if download_success and os.path.exists(zip_path):
        try:
            file_size = os.path.getsize(zip_path)
            print(f"📁 ZIP file size: {file_size / (1024*1024):.2f} MB")
            
            # Verify it's a valid zip file
            if zipfile.is_zipfile(zip_path):
                # Extract the ZIP file
                print("📦 Extracting ZIP file...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(datasets_dir)
                
                # Remove the ZIP file after extraction
                os.remove(zip_path)
                print(f"✅ Datasets extracted successfully to: {datasets_dir}")
                
                # List the contents to verify
                if os.path.exists(datasets_dir):
                    contents = os.listdir(datasets_dir)
                    print(f"📂 Extracted contents: {contents}")
            else:
                print("❌ Downloaded file is not a valid ZIP archive")
                print("💡 The file might be an HTML error page. Check the Google Drive link permissions.")
                
        except Exception as e:
            print(f"❌ Error during extraction: {str(e)}")
    else:
        print("❌ All download methods failed")
        print()
        print("📋 MANUAL DOWNLOAD REQUIRED")
        print("=" * 50)
        print("Please download manually:")
        print(f"1. Open: https://drive.google.com/file/d/{file_id}/view?usp=sharing")
        print("2. Click 'Download anyway' button (ignore the virus scan warning)")
        print("3. Save the file as 'datasets.zip' in your project folder")
        print("4. Extract it to create the ./datasets/ folder")
        print()
        print("💡 The virus scan warning is normal for large files and can be safely ignored")


if __name__ == "__main__":
    download_datasets()