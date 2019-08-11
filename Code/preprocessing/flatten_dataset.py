# The scraped images are sorted by Gundam show in subfolders, this script puts
# them all in a single folder.

import os
import shutil
from pathlib import Path



def main():

    input_folder_path = Path("../../Generated/scraping/downloaded_images/")
    output_folder_path = Path("../../Generated/preprocessing/flattened_images/")

    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path, exist_ok=True)

    for show_folder_path in input_folder_path.iterdir():
        for image_path in show_folder_path.iterdir():
            shutil.copy(image_path, output_folder_path)


if __name__ == "__main__":
    main()
