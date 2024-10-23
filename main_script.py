import os
import gdown

MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Define the function to download files
def download_all_files(file_ids):
    downloaded_files = {}
    for name, file_id in file_ids.items():
        file_extension = ".json" if "config" in name else ".pth"
        file_path = os.path.join(MODELS_DIR, f"{name}{file_extension}")

        if os.path.exists(file_path):
            downloaded_files[name] = file_path
            print(f"File already exists: {file_path}")
        else:
            gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
            downloaded_files[name] = file_path
            print(f"Downloaded: {file_path}")

    # Print status of all downloaded files
    print("All files processed:")
    for name, path in downloaded_files.items():
        print(f"{name}: {path}")

    return downloaded_files
    
def read_model_links(file_path):
    file_ids = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Assuming each line has the format: <name>:<file_id>
            name, file_id = line.strip().split(':')
            file_ids[name] = file_id
    return file_ids

# File IDs from Google Drive (use your pre-defined model links)
file_ids = read_model_links('model_links.txt')

# Download the files
downloaded_files = download_all_files(file_ids)

# Now that files are downloaded, run the FastAPI and Gradio-based app
os.system("python final_server.py")  # Call the FastAPI script to start the server

