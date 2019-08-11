# Downloads images from the scraped URLs, and saves them in different subfolders,
# sorted by Gundam show.

import os
import shutil
import json
from pathlib import Path
import urllib.request
from urlpath import URL
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) "

def main():

    input_url_list_filepath = Path("../../Generated/scraping/images_url.json")

    save_folder_path = Path("../../Generated/scraping/downloaded_images/")
    shutil.rmtree(save_folder_path, ignore_errors=True)
    os.makedirs(save_folder_path, exist_ok=True)

    with open(input_url_list_filepath, "r") as f:
        img_url_by_show = json.load(f)

    for show_name, img_url_list in img_url_by_show.items():
        print("Show :", show_name)
        saved_show_folder_path = save_folder_path / show_name
        os.makedirs(saved_show_folder_path, exist_ok=True)
        for img_url in img_url_list:
            _download_image(img_url, str(saved_show_folder_path))


def _download_image(img_url, folder_path):

    print("\tDownloading %s" % img_url)

    img_filename = URL(img_url).name

    req = urllib.request.Request(img_url, headers={'User-Agent': USER_AGENT})

    try:
        response = urllib.request.urlopen(req)
    except urllib.error.HTTPError:
        return

    output_filepath = Path(folder_path) / img_filename
    with open(str(output_filepath), "wb") as f:
        f.write(response.read())


if __name__ == "__main__":
    main()
