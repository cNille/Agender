from google_images_download import google_images_download  #importing the library
from os import listdir
from os.path import isfile, join

response = google_images_download.googleimagesdownload()  #class instantiation

# SEARCH = "en bildad person"
SEARCH = "doctor"

arguments = {
    "output_directory": './images',
    "keywords": SEARCH,
    "limit": 10,
    "print_urls": False
}  #creating list of arguments
response.download(arguments)  #passing the arguments to the function

mypath = './images/' + SEARCH
onlyfiles = [
    f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('jpg')
]

import prediction

agender = prediction.Agender()

for img in onlyfiles:
    agender.predict(mypath + '/' + img)
