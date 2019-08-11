# Scrapes the mahq.net website for mobile suit images and saves their URLs (sorted by show) to a JSON file.

import os
import re
from pathlib import Path
import json
import urllib.request
from urlpath import URL
from bs4 import BeautifulSoup

SITE_URL = "https://www.mahq.net/mecha/gundam/index.htm"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) "
IMG_FILENAMES_BLACKLIST = ["spacer.gif"]
MOBILE_SUIT_TYPES_PATTERN = "(mobile +suit)|(transformable +mobile +armor)|(mobile +fighter)"

def main():

    img_url_by_show = _scrape(SITE_URL)

    for key, img_url_list in img_url_by_show.items():
        print(key)
        for img_url in img_url_list:
            print("\t%s" % img_url)

    output_folder_path = Path("../../Generated/scraping/")
    os.makedirs(output_folder_path, exist_ok=True)
    output_filepath = output_folder_path / "images_url.json"
    with open(output_filepath, "w") as f:
        json.dump(img_url_by_show, f)

def _scrape(index_url):

    req = urllib.request.Request(index_url, headers={'User-Agent': USER_AGENT})

    try:
        response = urllib.request.urlopen(req)
    except urllib.error.HTTPError:
        return []

    soup = BeautifulSoup(response.read(), features="html.parser")

    img_url_by_show = {}
    for tag in soup.find_all("td", class_="shadow-horizontal"):
        for subtag in tag.find_all("a"):
            if "href" in subtag.attrs.keys():

                relative_show_url = URL(subtag["href"])
                show_url = URL(index_url).parent / relative_show_url

                show_name = show_url.parent.name
                img_url_by_show[show_name] = _scrape_show_page(str(show_url))

    return img_url_by_show


def _scrape_show_page(show_url):

    print("Scraping %s" % show_url)

    req = urllib.request.Request(show_url, headers={'User-Agent': USER_AGENT})

    try:
        response = urllib.request.urlopen(req)
    except urllib.error.HTTPError:
        return []

    soup = BeautifulSoup(response.read(), features="html.parser")

    img_url_lists = []
    for tag in soup.find_all("td", class_="shadow-horizontal"):
        for subtag in tag.find_all("a"):
            if "href" in subtag.attrs.keys():

                relative_suit_url = URL(subtag["href"])
                suit_url = URL(show_url).parent / relative_suit_url

                img_url_lists += _scrape_suit_page(str(suit_url))

    return img_url_lists

def _scrape_suit_page(suit_url):

    print("\tScraping %s" % suit_url)

    req = urllib.request.Request(suit_url, headers={'User-Agent': USER_AGENT})

    try:
        response = urllib.request.urlopen(req)
    except urllib.error.HTTPError:
        return []

    soup = BeautifulSoup(response.read(), features="html.parser")

    if not _is_mobile_suit(soup):
        print("\t\tNot a mobile suit, skipping...")
        return []

    # Getting absolute image URLs
    img_url_lists = []
    for img_tag in soup.find_all("img"):

        relative_img_url = URL(img_tag["src"])

        if relative_img_url.name in IMG_FILENAMES_BLACKLIST:
            continue

        img_url = URL(suit_url).parent / relative_img_url
        img_url_lists.append(str(img_url))

        break # Stopping at the first image

    return img_url_lists

def _is_mobile_suit(soup):

    for tag in soup.find_all("p"):
        if re.search("Unit +type", tag.text) and \
           re.search(MOBILE_SUIT_TYPES_PATTERN, tag.text):
            return True
    return False

if __name__ == "__main__":
    main()
