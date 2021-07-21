#!/usr/bin/env python
"""    
    .gitignore file creator - Justin Fuhrmeister-Clarke.
    Copyright (C) 2018  Justin Fuhrmeiser-Clarke <justin@fuhrmeister-clarke.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    """

import requests
import json
import sys

def get_json(path):
    return json.loads(get_text(path))
    
def get_text(path):
    return requests.get("https://api.github.com/repos/github/gitignore/contents/").text
    
#get list of gitignore files
def get_latest():
    files = {}
    titles = []
    files_json = get_json("https://api.github.com/repos/github/gitignore/contents/")
    for fileignore in files_json:
        titles.append(fileignore['name'])
        files[fileignore['name']]={
                'name': fileignore['name'],
                'url': fileignore['download_url'],
            }
    return {'files':files, 'titles':titles}

#download selected gitignore file
def get_file(title):
    #baseDir = https://raw.githubusercontent.com/github/gitignore/master/
    return requests.get("https://raw.githubusercontent.com/github/gitignore/master/{}".format(title)).text

#place into current directory
def save_file(title,text):
    f = open("./{}".format(title),"a")
    f.write(text)
    f.close()


#if no parameter appended, print all titles,
#else save file


if len(sys.argv)>=2 :
    title = sys.argv[1]
    print("Downloading {}".format(title))
    text = get_file(title)
    print("Saving to .gitignore")
    save_file('.gitignore',text)
    
else:
    print("Usage {} <file.title>".format(sys.argv[0]))
    print("")
    for title in get_latest()['titles']:
        print(title)
    print("")
    print("Usage {} <file.title>".format(sys.argv[0]))
    print("")

