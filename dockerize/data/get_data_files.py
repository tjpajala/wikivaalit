from zipfile import ZipFile
import urllib.request
from tqdm import tqdm
import os

def my_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


#load yle data
url = "https://vaalit.beta.yle.fi/avoindata/candidate_answer_data_kokomaa11042017.zip"
local_zip_name = "./candidate_answer_data_kokomaa11042017.zip"
if local_zip_name.split(".")[1][1:]+".csv" not in os.listdir():
    with tqdm(...) as t:
        reporthook = my_hook(t)
        urllib.request.urlretrieve(url, "candidate_answer_data_kokomaa11042017.zip", reporthook)
    zip_ref = ZipFile(local_zip_name, 'r')
    zip_ref.extractall(".")
    zip_ref.close()

#load wikipedia data
url_wiki = "https://dumps.wikimedia.org/fiwiki/20190101/fiwiki-20190101-pages-articles.xml.bz2"
#local_wiki_name = "./fiwiki-20190101-pages-articles.xml.bz2"
if "fiwiki-20190101-pages-articles.xml.bz2" not in os.listdir():
    with tqdm(...) as t:
        reporthook = my_hook(t)
        urllib.request.urlretrieve(url_wiki, "fiwiki-20190101-pages-articles.xml.bz2", reporthook)