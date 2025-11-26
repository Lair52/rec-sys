import zipfile
import os


def unzip_data():
    zip_path = "data.zip"
    extract_to = "data/"

    # Create folder if not exists
    os.makedirs(extract_to, exist_ok=True)

    # Unzip data.zip
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    print("File saved in:", extract_to)


if __name__ == "__main__":
    unzip_data()
