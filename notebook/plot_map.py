from itertools import combinations
import os.path
from pathlib import Path
import stat
from sys import platform

import pandas as pd
import folium

import selenium.webdriver
import requests

CASE_NAME='test_2015_2018'
def read_csv():
    station_csv = "/mnt/post/station/station_stats.csv"
    station_csv = '/home/appleparan/data/' + CASE_NAME + '/post/station/station_stats.csv'

    map_dir = "./"
    df = pd.read_csv(station_csv)

    return df 

def draw_map(df, ycol, evalname, bins, fname):
    """
        lat = float(row[2])
        lon = float(row[3])
        name = row[1] 
        RSR = row[6]
        corr_24h = row[11]
    """

    seoul = (37.55873, 127.01912)
    base_path = '/home/appleparan/data/' + CASE_NAME
    map_dir = Path(base_path) / 'map'
    print(map_dir)
    if not map_dir.is_dir() and not map_dir.exists():
        map_dir.mkdir()
    map_seoul = folium.Map(location=seoul,
                           zoom_start=11)
    corr_bins = [0.5,0.6,0.7,0.8,0.9,1]
    RSR_bins = [0.0,0.1,0.2,0.3,0.4,0.5]

    _bins = corr_bins
    _color = 'PuBu'
    if evalname == 'RSR':
        _bins = RSR_bins
        _color = 'PuBu_r'
    elif evalname == 'corr_24h' or evalname == 'corr_01h':
        _bins = corr_bins

    geo_path = '/home/appleparan/src/MLToys.jl/notebook/skorea-municipalities-2018-geo.json'
    geo_path = '/home/appleparan/src/MLToys.jl/notebook/seoul_municipalities_geo.json'
    df_ycol = df[df['colname'] == ycol]
    df_map = df_ycol[['name', evalname]]

    folium.Choropleth(
        geo_data = geo_path,
        data=df_map,
        columns=['name', evalname],
        key_on='feature.properties.SIG_KOR_NM',
        fill_color=_color,
        fill_opacity=0.7,
        line_opacity=0.5,
        bins = bins,
        legend_name=evalname + ' on ' + ycol,
        control=False, reset=True).add_to(map_seoul)
    map_seoul.save(str(map_dir / 'map.html'))

    map2png(map_dir, fname)

def map2png(map_dir, fname):
    chrome_dirname = 'chrome_drv'
    # create chrome
    chrome_dir = map_dir / chrome_dirname
    chrome_dir.mkdir(exist_ok=True)

    chrome_path = Path(chrome_dir / 'chromedriver')
    if platform == "win32":
        chrome_path = Path(chrome_dir / 'chromedriver.exe')

    if not chrome_path.exists():
        download_chromedriver(chrome_dir, chrome_path)

    options = selenium.webdriver.ChromeOptions()
    options.add_argument('headless')
    try:
        driver = selenium.webdriver.Chrome(
            chrome_options=options,
            executable_path=str(chrome_path))
    except:
        print("Selenium Error")
        raise
    else:
        driver.set_window_size(1080, 720)  # choose a resolution
        driver.get(Path(map_dir / 'map.html').resolve().as_uri())

        # You may need to add time.sleep(seconds) here
        import time
        time.sleep(2)

        #cwd = os.getcwd()
        # abs_path = os.path.abspath(cwd)
        driver.save_screenshot(str(map_dir / str(fname + '.png')))
        driver.quit()


def download_chromedriver(chrome_dir, chrome_path):
    """
        download chromedriver and unzip to chrome_dir
        https://www.codementor.io/aviaryan/downloading-files-from-urls-in-python-77q3bs0un
    :param chrome_dir:
    :param chrome_path:
    :return:
    """

    chromedrv_ver = requests.get('https://chromedriver.storage.googleapis.com/LATEST_RELEASE').content.decode("utf-8")
    if platform == "linux":
        filename = 'chromedriver_linux64'
        url = 'https://chromedriver.storage.googleapis.com/' + chromedrv_ver + '/' + filename + '.zip'
    elif platform == "darwin":
        filename = 'chromedriver_mac64'
        url = 'https://chromedriver.storage.googleapis.com/' + chromedrv_ver + '/' + filename + '.zip'
    elif platform == "win32":
        filename = 'chromedriver_win32'
        url = 'https://chromedriver.storage.googleapis.com/'  + chromedrv_ver + '/' + filename + '.zip'
    else:
        raise ValueError('Invalid system platform: Not in Linux, macOS, Windows')

    r = requests.get(url, allow_redirects=True)
    #filename = get_filename_from_cd(r.headers.get('content-disposition'))
    open(str(chrome_dir / str(filename + '.zip')), 'wb').write(r.content)

    import zipfile

    zip_ref = zipfile.ZipFile(str(chrome_dir / str(filename + '.zip')), 'r')
    zip_ref.extractall(str(chrome_dir))
    zip_ref.close()

    if platform != 'win32':
        os.chmod(str(chrome_path), stat.S_IEXEC)


def run():
    df = read_csv()

    #draw_map(df, 'PM10', 'RSR', 'seoul_RSR_PM10')
    #draw_map(df, 'PM25', 'RSR', 'seoul_RSR_PM25')
    corr_bins = [0.0,0.2,0.4,0.6,0.8,1]
    draw_map(df, 'PM10', 'corr_24h', corr_bins, 'seoul_corr_24h_PM10')
    corr_bins = [0.5,0.6,0.7,0.8,0.9,1]
    draw_map(df, 'PM25', 'corr_24h', corr_bins, 'seoul_corr_24h_PM25')


if __name__ == '__main__':
    run()
