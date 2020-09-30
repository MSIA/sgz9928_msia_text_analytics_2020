# HW-1

**Name:** Ganguly, Shreyashi

**NetID:** sgz9928

GitHub link - https://github.com/MSIA/sgz9928_msia_text_analytics_2020.git
Branch - homework1

## Problem 1

The following python NLP libraries were compared:
- nltk
- spacy
- stanza

The text corpus chosen to compare the libraries:
- 20_newsgroups - talk.religion.misc
- 997 files were read
- total corpus size = 2.65 MB

Each library was compared based on the operations tokenization, stemming/lemmatizationa and POS tagging. <br>
Various aspects like run time, memory usage, ease of use etc. were observed to arrive at a fair comparison. <br>

The results of the exercise are summarised as follows: <br>
|  Aspect   |  Operation   |  NLTK  |  Spacy  |  Stanza  |
| --- | --- | --- | --- | --- |
|  Runtime   |  word tokenize  |  5.685s  |  1.897s  |  0.123s  |
|  Runtime   |  stemming lemmatization  |  8.624s  |  1.783s  |  0.128s  |
|  Runtime   |  POS tagging  |  19.558s  |  0.679s  |  0.112s  |
|  Memory Usage   |  average  |  119.6MB  |  435.8MB  |  580.7MB |
|  Memory Usage   |  peak  |  141.8MB  |  10155.7MB  |  1353.3MB |

<br>
- As can be inferred from the above table, NLTK requires slightly more time than the other two libraries <br>
- NLTK however consumes much less memory than the others <br>
- NLTK feels the most intuitive with different methods for word tokenization, sentence parsing, stemming, lemmatization, POS tagging etc. <br>
- Spacy could not process more than 1 million characters by default. The nlp.max_length parameter had to be reset to process the text corpus <br>
- Spacy allowed for all the steps of creating tokens, lemmas and POS tags to be completed in one step <br>
- Stanza's syntax for sentence parsing is complicated and counter productive <br>
- The lemmas obtained from Stanza also seemed counter intuitive with words like biased, highly retained as is instead of being reduced to bias and high
- Loading the data in the pipeline took a lot of time which is not accounted for in the runtimes provided in the above table. This however, felt like a major drawback. <br>

<br>
In conclusion, from my experience, I would prefer to work with NLTK and Spacy

### Relevant Files

- Python script can be found [here](https://github.com/MSIA/sgz9928_msia_text_analytics_2020/blob/homework1/problem1.py)
- Run time and performance measures can be found [here](https://github.com/MSIA/sgz9928_msia_text_analytics_2020/blob/homework1/nlp_library_comparison.txt)


## Problem 2

Number of files parsed = 995 <br>
Total size of text corpus = 2171133 bytes

Some of the different formats of dates parsed as as follows: <br>
'18 Apr 1993' <br>
'22 Apr 1993' <br>
'16 Apr 93' <br>
'2 May 19' <br>
'1993/04/27' <br>
'1993/04/17'

Some of the different formats of emails parsed are as follows: <br>
'danny.hawrysio@canrem.com' <br>
'ch381@cleveland.Freenet.Edu' <br>
'z_nixsp@ccsvax.sfasu.edu' <br>
'1993Apr15.155849.22753@ghost.dsi.unimi.it'

The entire list can be found in the github repo

### Relevant Files

- Python script can be found [here](https://github.com/MSIA/sgz9928_msia_text_analytics_2020/blob/homework1/problem2.py)
- List of parsed dates and emails can be found [here](https://github.com/MSIA/sgz9928_msia_text_analytics_2020/blob/homework1/regex_dates_email.txt)




