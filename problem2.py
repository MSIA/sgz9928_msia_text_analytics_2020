import os
from pathlib import Path
import re

# Reading in files from folder 'talk.religious.misc'
mypath = "/Users/shreyashiganguly/Documents/Northwestern_MSiA/Fall2020/Text_Analytics/HW1/20_newsgroups/comp.graphics"

def read_file(filename, mypath=mypath):
    filepath = os.path.join(mypath, filename)
    filesize = os.path.getsize(filepath)
    text = Path(filepath).read_text()
    return filesize, text

if __name__ == '__main__':
    f = open('regex_dates_email.txt', 'w')
    text_corpus = ""
    num_files = 0
    file_size = 0
    list_of_files = os.listdir(mypath)
    for filename in list_of_files:
        try:
            size, text = read_file(filename)
            text_corpus = text_corpus + text
            num_files += 1
            file_size += size
        except UnicodeDecodeError as e:
            print(f'Skipping {filename} due to Unicode error')

    print(f"Number of files parsed = {num_files}", file=f)
    print(f"Total size of text corpus = {file_size} bytes", file=f)

    # Finding all dates
    all_dates = []

    dates = re.findall(r"[\d]{1,2} [ADFJMNOS]\w* [\d]{4}", text_corpus)     # matching dates in format 10 OCT 2015 or 10 Oct 2015 or 10 October 2015
    for each in dates:
        all_dates.append(each)

    dates = re.findall(r"[ADFJMNOS]\w* [\d]{1,2}(th){0,1} [\d]{4}", text_corpus)     # matching dates in format OCT 10 2015 or Oct 10th 2015 or October 10 2015
    for each in dates:
        all_dates.append(each)

    dates = re.findall(r"[\d]{1,2} [ADFJMNOS]\w* [\d]{2}", text_corpus)     # matching dates in format 10 OCT 15 or 10 Oct 15 or 10 October 15
    for each in dates:
        all_dates.append(each)

    dates = re.findall(r"\b[\d]{1,2}/[\d]{1,2}/[\d]{4}\b", text_corpus)     # matching dates in format 10/10/2015
    for each in dates:
        all_dates.append(each)

    dates = re.findall(r"\b[\d]{4}-[\d]{1,2}-[\d]{1,2}\b", text_corpus)     # matching dates in format 10-10-2015
    for each in dates:
        all_dates.append(each)

    dates = re.findall(r"\b[\d]{4}/[\d]{1,2}/[\d]{1,2}\b", text_corpus)     # matching dates in format 2015/10/10
    for each in dates:
        all_dates.append(each)

    dates = re.findall(r"\b[\d]{4}-[\d]{1,2}-[\d]{1,2}\b", text_corpus)     # matching dates in format 2015-10-10
    for each in dates:
        all_dates.append(each)

    print(f'\n\nAll parsed dates:\n {all_dates}', file=f)


    # Find all email ids
    all_email = []

    mail = re.findall(r"([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)", text_corpus)     # matching dates in format 10 OCT 2015 or 10 Oct 2015 or 10 October 2015
    for each in mail:
        all_email.append(each)

    print(f'\n\nAll parsed emails:\n {all_email}', file=f)

    f.close()




