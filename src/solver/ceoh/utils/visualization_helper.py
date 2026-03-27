# Copyright (c) 2025
#           Thomas Bömer (thomas.bömer@tu-dortmund.de)
#           Nico Koltermann (nico.koltermann@tu-dortmund.de) 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import re
import json

def insert_line_breaks(text, max_length=50):
    """ Inserts <br> into the text if a line exceeds max_length, ensuring no words are split. """
    words = text.split()
    line = ""
    formatted_text = ""

    for word in words:
        if len(line) + len(word) + 1 > max_length:
            formatted_text += line.strip() + "<br>"
            line = word + " "
        else:
            line += word + " "

    formatted_text += line.strip()
    return formatted_text


def load_all_data(path, sort_before=False):

    file_names = os.listdir(path)
    if sort_before:
        file_names = sorted(
            [f for f in os.listdir(path) if f.endswith('.json')],
            key=lambda x: int(re.search(r'\d+', x).group())
        )

    files = []

    for filename in file_names:
        if filename.endswith(".json"):
            file_path = os.path.join(path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    temp = {
                        'filename': filename,
                        'programs': data
                    }
                    files.append(temp)
            except:
                print("Failed")

    return files


def get_by_id(data, key):
    for d in data:
        if d['programs']['offspring']['offspring_id'] == key:
            return d

