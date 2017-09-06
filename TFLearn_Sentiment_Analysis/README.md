
# Sentiment analysis with TFLearn

In this notebook, we'll continue Andrew Trask's work by building a network for sentiment analysis on the movie review data. Instead of a network written with Numpy, we'll be using [TFLearn](http://tflearn.org/), a high-level library built on top of TensorFlow. TFLearn makes it simpler to build networks just by defining the layers. It takes care of most of the details for you.

We'll start off by importing all the modules we'll need, then load and prepare the data.


```python
import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
```

## Preparing the data

Following along with Andrew, our goal here is to convert our reviews into word vectors. The word vectors will have elements representing words in the total vocabulary. If the second position represents the word 'the', for each review we'll count up the number of times 'the' appears in the text and set the second position to that count. I'll show you examples as we build the input data from the reviews data. Check out Andrew's notebook and video for more about this.

### Read the data

Use the pandas library to read the reviews and postive/negative labels from comma-separated files. The data we're using has already been preprocessed a bit and we know it uses only lower case characters. If we were working from raw data, where we didn't know it was all lower case, we would want to add a step here to convert it. That's so we treat different variations of the same word, like `The`, `the`, and `THE`, all the same way.


```python
reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)
```

### Counting word frequency

To start off we'll need to count how often each word appears in the data. We'll use this count to create a vocabulary we'll use to encode the review data. This resulting count is known as a [bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model). We'll use it to select our vocabulary and build the word vectors. You should have seen how to do this in Andrew's lesson. Try to implement it here using the [Counter class](https://docs.python.org/2/library/collections.html#collections.Counter).

> **Exercise:** Create the bag of words from the reviews data and assign it to `total_counts`. The reviews are stores in the `reviews` [Pandas DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html). If you want the reviews as a Numpy array, use `reviews.values`. You can iterate through the rows in the DataFrame with `for idx, row in reviews.iterrows():` ([documentation](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iterrows.html)). When you break up the reviews into words, use `.split(' ')` instead of `.split()` so your results match ours.


```python
reviews
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bromwell high is a cartoon comedy . it ran at ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>story of a man who has unnatural feelings for ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>homelessness  or houselessness as george carli...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>airport    starts as a brand new luxury    pla...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>brilliant over  acting by lesley ann warren . ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>this film lacked something i couldn  t put my ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>this is easily the most underrated film inn th...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sorry everyone    i know this is supposed to b...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>this is not the typical mel brooks film . it w...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>when i was little my parents took me along to ...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>this isn  t the comedic robin williams  nor is...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>it appears that many critics find the idea o...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>yes its an art . . . to successfully make a sl...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>the second attempt by a new york intellectual ...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>in this  critically acclaimed psychological th...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>i don  t know who to blame  the timid writers ...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>the night listener           robin williams  t...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>this film is mediocre at best . angie harmon i...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>you know  robin williams  god bless him  is co...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>the film is bad . there is no other way to say...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>when i first read armistead maupins story i wa...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>this film is one giant pant load . paul schrad...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>i liked the film . some of the action scenes w...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>the plot for descent  if it actually can be ca...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>there are many illnesses born in the mind of m...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>plot is not worth discussion even if it hints ...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>i enjoyed the night listener very much . it  s...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>this film is about a male escort getting invol...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>the night listener is probably not one of will...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>this movie must be in line for the most boring...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>24970</th>
      <td>i had a chance to see a screening of this movi...</td>
    </tr>
    <tr>
      <th>24971</th>
      <td>i had the misfortune to watch this rubbish on ...</td>
    </tr>
    <tr>
      <th>24972</th>
      <td>this is a really interesting movie . it is an ...</td>
    </tr>
    <tr>
      <th>24973</th>
      <td>it  s pretty bad when the generic movie synops...</td>
    </tr>
    <tr>
      <th>24974</th>
      <td>i saw the movie recently and really liked it ....</td>
    </tr>
    <tr>
      <th>24975</th>
      <td>having watched this movie on the scifi channel...</td>
    </tr>
    <tr>
      <th>24976</th>
      <td>i thought this movie was hysterical . i have w...</td>
    </tr>
    <tr>
      <th>24977</th>
      <td>first off  i  m not here to dog this movie . i...</td>
    </tr>
    <tr>
      <th>24978</th>
      <td>. . . this is a classic with so many great di...</td>
    </tr>
    <tr>
      <th>24979</th>
      <td>ah yez  the sci fi channel produces yeti anoth...</td>
    </tr>
    <tr>
      <th>24980</th>
      <td>the most hillarious and funny brooks movie i e...</td>
    </tr>
    <tr>
      <th>24981</th>
      <td>yeti curse of the snow demon starts aboard a p...</td>
    </tr>
    <tr>
      <th>24982</th>
      <td>life stinks  is a parody of life and death  ...</td>
    </tr>
    <tr>
      <th>24983</th>
      <td>hmmm  a sports team is in a plane crash  gets ...</td>
    </tr>
    <tr>
      <th>24984</th>
      <td>this is the kind of film you want to see with ...</td>
    </tr>
    <tr>
      <th>24985</th>
      <td>i saw this piece of garbage on amc last night ...</td>
    </tr>
    <tr>
      <th>24986</th>
      <td>i have not read the other comments on the film...</td>
    </tr>
    <tr>
      <th>24987</th>
      <td>although the production and jerry jameson  s d...</td>
    </tr>
    <tr>
      <th>24988</th>
      <td>life stinks       was a step below mel brooks ...</td>
    </tr>
    <tr>
      <th>24989</th>
      <td>capt . gallagher  lemmon  and flight attendant...</td>
    </tr>
    <tr>
      <th>24990</th>
      <td>seeing as the vote average was pretty low  and...</td>
    </tr>
    <tr>
      <th>24991</th>
      <td>towards the end of the movie  i felt it was to...</td>
    </tr>
    <tr>
      <th>24992</th>
      <td>the plot had some wretched  unbelievable twist...</td>
    </tr>
    <tr>
      <th>24993</th>
      <td>this is the kind of movie that my enemies cont...</td>
    </tr>
    <tr>
      <th>24994</th>
      <td>i am amazed at how this movie  and most others...</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>i saw  descent  last night at the stockholm fi...</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>a christmas together actually came before my t...</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>some films that you pick up for a pound turn o...</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>working  class romantic drama from director ma...</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>this is one of the dumbest films  i  ve ever s...</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 1 columns</p>
</div>




```python
import types
type(reviews)
```




    pandas.core.frame.DataFrame



from collections import Counter

reviews = reviews.T
total_counts_set = set()# bag of words here
for review in reviews[0]:
    for word in review.split(" "):
        total_counts_set.add(word)
total_counts_set = list(total_counts_set)
"""        
total_counts = Counter()
for i in range(len(total_counts_set)):
    word = total_counts_set[i]
    for j in range(len(total_counts_set)):
        if (word == total_counts_set[j]):            
            total_counts[word] += 1

"""

print("Total words in data set: ", len(total_counts_set))
#print(total_counts_set.type)

total_counts = Counter()
for i in range(len(total_counts_set)):
    word = total_counts_set[i]
    for j in range(len(total_counts_set)):
        if (word == total_counts_set[j]):            
            total_counts[word] += 1

#total_counts_set
total_counts


```python
from collections import Counter
#reviews = reviews.T
total_counts = Counter()# bag of words here
for _ , row in reviews.iterrows():
    total_counts.update(row[0].split(' '))
print("Total words in data set: ", len(total_counts))
```

    Total words in data set:  74074



```python
total_counts
```




    Counter({'': 1111930,
             'brielfy': 1,
             'soaring': 11,
             'patronization': 1,
             'aleck': 6,
             'dena': 1,
             'epoque': 1,
             'purchases': 8,
             'catalua': 1,
             'rabbitt': 1,
             'volckman': 18,
             'marc': 79,
             'biting': 59,
             'results': 274,
             'red': 818,
             'diverts': 5,
             'juicy': 26,
             'target': 210,
             'glistening': 1,
             'closer': 206,
             'tirelli': 2,
             'studies': 70,
             'garment': 3,
             'alejandra': 3,
             'statistics': 13,
             'newsradio': 1,
             'resnais': 18,
             'girlfirend': 1,
             'electrocutes': 6,
             'flavorful': 3,
             'diss': 1,
             'leung': 18,
             'litle': 1,
             'antibiotics': 1,
             'champmathieu': 1,
             'mcnairy': 1,
             'mortgan': 1,
             'drawer': 20,
             'jorgen': 4,
             'courts': 15,
             'linked': 49,
             'elmer': 37,
             'towner': 4,
             'sita': 1,
             'herbie': 18,
             'lucia': 8,
             'itwould': 1,
             'lacks': 365,
             'occasions': 75,
             'draine': 1,
             'sensitively': 16,
             'notti': 2,
             'bolt': 23,
             'blueprint': 1,
             'scrabble': 1,
             'fastforward': 2,
             'tether': 2,
             'tftc': 6,
             'disase': 1,
             'recoil': 7,
             'pregnancies': 2,
             'qualms': 17,
             'flipped': 16,
             'elicot': 6,
             'fectly': 1,
             'demarcation': 1,
             'presses': 7,
             'velde': 3,
             'freeloader': 1,
             'predilection': 6,
             'valueless': 2,
             'annisten': 1,
             'everyplace': 1,
             'grannies': 3,
             'tra': 3,
             'speeders': 5,
             'yeop': 1,
             'saranden': 1,
             'frogging': 1,
             'benefits': 48,
             'richmont': 1,
             'pleasures': 38,
             'benkai': 1,
             'sartor': 1,
             'speed': 249,
             'bludgeoning': 7,
             'zombified': 9,
             'ductwork': 1,
             'verhooven': 1,
             'jameson': 34,
             'doodads': 1,
             'seachange': 1,
             'apalled': 1,
             'lodz': 1,
             'demerit': 2,
             'gondry': 1,
             'reignite': 4,
             'placed': 188,
             'ridd': 1,
             'removal': 17,
             'halperins': 2,
             'westway': 2,
             'wraparound': 5,
             'invites': 75,
             'knuckleheads': 2,
             'caricaturish': 1,
             'prepaid': 1,
             'oppikoppi': 1,
             'horrormovie': 1,
             'boffo': 4,
             'ceases': 19,
             'commercialism': 9,
             'mertz': 1,
             'blalock': 7,
             'catalysis': 1,
             'swathes': 3,
             'sharman': 5,
             'foil': 50,
             'unwatched': 7,
             'helped': 324,
             'syringe': 3,
             'reliance': 20,
             'resplendent': 2,
             'leone': 26,
             'cuts': 274,
             'treacherously': 1,
             'exult': 1,
             'adults': 376,
             'swoop': 10,
             'blander': 3,
             'mammonist': 1,
             'alphaville': 3,
             'molt': 1,
             'substantiate': 1,
             'infuriate': 2,
             'denom': 1,
             'stroll': 17,
             'plaster': 6,
             'darth': 52,
             'iron': 94,
             'coincidentally': 28,
             'kimbo': 1,
             'beetlejuice': 6,
             'salted': 1,
             'pan': 103,
             'babbitt': 1,
             'distraction': 54,
             'crisping': 1,
             'nirgendwo': 1,
             'defied': 9,
             'curiousness': 1,
             'punchier': 2,
             'cuban': 54,
             'jhurhad': 1,
             'repentance': 6,
             'poochie': 2,
             'devises': 3,
             'tijuana': 2,
             'bosh': 1,
             'pervert': 40,
             'backbone': 27,
             'broadways': 1,
             'qi': 11,
             'brogue': 5,
             'melted': 11,
             'leticia': 2,
             'mallrats': 2,
             'broadened': 1,
             'austens': 4,
             'primrose': 1,
             'point': 3224,
             'stockler': 6,
             'expending': 2,
             'mullets': 7,
             'marilee': 1,
             'nobodies': 10,
             'consummation': 1,
             'roofs': 10,
             'mugs': 9,
             'penn': 76,
             'poelzig': 1,
             'conquest': 36,
             'hog': 19,
             'superstition': 4,
             'spectecular': 1,
             'talman': 1,
             'questions': 479,
             'jayaraj': 1,
             'edmund': 43,
             'are': 29430,
             'picturesquely': 1,
             'beart': 3,
             'bereft': 11,
             'hugues': 1,
             'adair': 2,
             'chasey': 11,
             'brainy': 11,
             'minha': 2,
             'lexus': 1,
             'misspent': 2,
             'stoltz': 34,
             'kuba': 1,
             'expel': 2,
             'reenberg': 2,
             'gnome': 4,
             'snotty': 14,
             'despot': 2,
             'oceans': 13,
             'galore': 30,
             'quigley': 18,
             'hagarty': 2,
             'ineptness': 12,
             'commercialization': 4,
             'politburo': 1,
             'pyar': 11,
             'warships': 2,
             'uncharted': 12,
             'boating': 3,
             'majo': 1,
             'terminatrix': 1,
             'male': 666,
             'nubile': 19,
             'punching': 25,
             'immatured': 1,
             'oe': 1,
             'choreographer': 34,
             'parent': 119,
             'gauleiter': 1,
             'whereupon': 6,
             'klingons': 8,
             'salesmanship': 1,
             'masterly': 5,
             'historic': 73,
             'serat': 2,
             'fernandez': 11,
             'proxy': 9,
             'drainingly': 1,
             'readout': 2,
             'tamakwa': 3,
             'uncomfortably': 19,
             'pabulum': 2,
             'accounted': 10,
             'broadsword': 1,
             'slomo': 2,
             'septuplets': 1,
             'shohei': 10,
             'hudson': 141,
             'engages': 23,
             'whocoincidentally': 1,
             'roebuck': 5,
             'collar': 47,
             'redeem': 69,
             'vandyke': 1,
             'blackploitation': 1,
             'ziering': 10,
             'firefall': 1,
             'sean': 262,
             'robt': 1,
             'dogging': 1,
             'guiol': 2,
             'bedroom': 107,
             'driller': 3,
             'jouissance': 7,
             'cozies': 1,
             'shatters': 3,
             'pastiches': 5,
             'bono': 10,
             'blocker': 10,
             'vivienne': 2,
             'karamchand': 4,
             'wat': 5,
             'irak': 5,
             'shshshs': 1,
             'queensland': 3,
             'walbrook': 3,
             'linens': 1,
             'lieber': 1,
             'melinda': 38,
             'globalization': 14,
             'zouzou': 1,
             'society': 676,
             'samples': 9,
             'factoring': 1,
             'rerunning': 2,
             'plainly': 21,
             'feathering': 2,
             'ruben': 14,
             'scoop': 68,
             'entitlements': 1,
             'viviane': 1,
             'pillow': 21,
             'unfussy': 1,
             'masterpiece': 612,
             'apply': 55,
             'ridiculousness': 20,
             'vosen': 4,
             'bicarbonate': 2,
             'stylistic': 29,
             'hypothermia': 4,
             'whored': 3,
             'indolent': 4,
             'reshipping': 1,
             'lessened': 8,
             'creaks': 7,
             'germane': 2,
             'iarritu': 1,
             'sevilla': 1,
             'reaching': 95,
             'seldomely': 1,
             'hoofing': 4,
             'fleischer': 18,
             'pes': 2,
             'palestine': 10,
             'derangement': 3,
             'sneaks': 25,
             'fishburn': 4,
             'performers': 150,
             'bhodi': 3,
             'van': 495,
             'philipp': 1,
             'mcgarrett': 2,
             'reanimates': 2,
             'dohhh': 1,
             'bearand': 1,
             'coalville': 1,
             'chowdhry': 1,
             'potions': 2,
             'sugden': 2,
             'moh': 12,
             'histrionic': 16,
             'kingship': 1,
             'debut': 260,
             'blockades': 1,
             'candela': 1,
             'scientifically': 1,
             'spewings': 1,
             'grumpier': 1,
             'striker': 1,
             'zeme': 1,
             'seargent': 1,
             'ariana': 1,
             'onwards': 22,
             'americanised': 3,
             'practitioners': 3,
             'keeper': 37,
             'demotes': 3,
             'guide': 125,
             'advisors': 5,
             'consolidated': 1,
             'automated': 3,
             'beverages': 9,
             'impossibly': 24,
             'icebergs': 3,
             'disturbed': 108,
             'readin': 2,
             'officious': 3,
             'fightm': 1,
             'nazgul': 6,
             'eritated': 1,
             'kewpie': 4,
             'starving': 31,
             'izoo': 1,
             'chojnacki': 1,
             'wormy': 2,
             'delusions': 19,
             'reacted': 13,
             'agendas': 10,
             'giddiness': 1,
             'antelopes': 1,
             'gratefully': 9,
             'griping': 4,
             'poisoner': 1,
             'resurrect': 24,
             'unreliable': 5,
             'soupon': 1,
             'conjugal': 4,
             'ninety': 46,
             'chaplinesque': 2,
             'itelf': 1,
             'macarena': 3,
             'nehru': 1,
             'plangent': 1,
             'paradice': 1,
             'simn': 1,
             'bricked': 2,
             'maerose': 1,
             'nicolae': 1,
             'chrouching': 1,
             'khrushchev': 2,
             'brash': 29,
             'safeguard': 1,
             'gitai': 4,
             'sellick': 1,
             'imposture': 1,
             'pollak': 18,
             'bouffant': 1,
             'wrote': 574,
             'opened': 155,
             'whorde': 1,
             'segonzac': 1,
             'crudeness': 7,
             'chores': 18,
             'samoans': 3,
             'darcy': 12,
             'fossilized': 1,
             'overconfident': 4,
             'collectors': 16,
             'neville': 16,
             'investogate': 1,
             'delivers': 356,
             'forrester': 9,
             'explanations': 37,
             'mensonges': 5,
             'destabilize': 1,
             'fluffee': 1,
             'commendation': 1,
             'gunned': 14,
             'stress': 98,
             'vulcans': 6,
             'adaptor': 1,
             'implode': 3,
             'arnett': 3,
             'plo': 1,
             'heinously': 2,
             'safeauto': 1,
             'pursestrings': 1,
             'aznable': 1,
             'quickest': 5,
             'kittenishly': 1,
             'baggy': 6,
             'frickin': 5,
             'minmay': 1,
             'hereby': 3,
             'amatuerish': 2,
             'retardation': 4,
             'emancipator': 3,
             'armors': 1,
             'hill': 243,
             'passes': 106,
             'treadstone': 3,
             'torment': 41,
             'dulany': 1,
             'duties': 33,
             'wald': 1,
             'jingles': 1,
             'contestant': 45,
             'mcneice': 1,
             'videotaping': 4,
             'comatose': 24,
             'flippantly': 1,
             'hai': 19,
             'dik': 1,
             'cellphone': 10,
             'soak': 7,
             'bright': 273,
             'motel': 45,
             'owes': 62,
             'cratchitt': 1,
             'boobage': 1,
             'dumpsters': 2,
             'caballeros': 1,
             'moko': 1,
             'nam': 20,
             'earthlings': 6,
             'brangelina': 1,
             'unnattractive': 2,
             'tellytubbies': 1,
             'aqua': 6,
             'morel': 1,
             'percept': 1,
             'ensigns': 1,
             'yeast': 1,
             'mood': 432,
             'shawl': 7,
             'leeched': 1,
             'tremble': 4,
             'dekhne': 3,
             'holroyd': 6,
             'cannae': 1,
             'sop': 2,
             'equivocal': 1,
             'nonlinear': 1,
             'duccio': 4,
             'tanked': 7,
             'ghajini': 4,
             'travolta': 43,
             'mallepa': 5,
             'petron': 1,
             'gerschwin': 1,
             'lowest': 92,
             'faulted': 7,
             'viceversa': 1,
             'nacio': 1,
             'lamma': 1,
             'borowczyk': 14,
             'duuum': 1,
             'farthest': 2,
             'vapoorize': 3,
             'legioners': 1,
             'woods': 400,
             'tunic': 3,
             'polygram': 1,
             'fictionalization': 5,
             'maliciously': 1,
             'blurt': 2,
             'klavan': 2,
             'clog': 1,
             'anal': 26,
             'generational': 10,
             'earlier': 663,
             'screenings': 17,
             'fiilthy': 1,
             'condemning': 15,
             'madolyn': 3,
             'breadth': 8,
             'complying': 3,
             'kapoor': 113,
             'kwrice': 1,
             'annen': 1,
             'roofthooft': 1,
             'witted': 37,
             'statesmanship': 1,
             'eeeevil': 1,
             'snares': 2,
             'weariness': 6,
             'counterstrike': 1,
             'mareno': 1,
             'wishing': 85,
             'orientalist': 1,
             'hurdles': 3,
             'bannen': 8,
             'hywel': 1,
             'whether': 856,
             'revelers': 6,
             'twentyish': 1,
             'sushmita': 8,
             'payal': 2,
             'playwriting': 1,
             'blabber': 2,
             'glassy': 3,
             'sr': 33,
             'istvan': 2,
             'cassamoor': 1,
             'stunning': 407,
             'fooling': 19,
             'bhaer': 1,
             'anothwer': 1,
             'canerday': 1,
             'nurturing': 7,
             'destines': 1,
             'bruhl': 11,
             'woronow': 1,
             'vivek': 4,
             'trenchant': 5,
             'sedan': 4,
             'contenders': 6,
             'boisterously': 2,
             'renaming': 1,
             'while': 5317,
             'employing': 14,
             'atmospheres': 6,
             'considers': 44,
             'tuaregs': 1,
             'evoking': 20,
             'glynn': 3,
             'laude': 4,
             'papamichael': 3,
             'ornery': 5,
             'subjectiveness': 1,
             'unwatchable': 107,
             'sociopathic': 8,
             'snobbery': 10,
             'creegan': 1,
             'microphone': 22,
             'staked': 2,
             'disproportionately': 4,
             'herringbone': 1,
             'vartan': 11,
             'seclusion': 4,
             'surprising': 302,
             'rahxephon': 1,
             'novotna': 2,
             'dove': 16,
             'devastatingly': 6,
             'revisioning': 1,
             'hailed': 24,
             'latine': 1,
             'cursed': 41,
             'vic': 34,
             'glide': 5,
             'rid': 118,
             'yearn': 18,
             'mtm': 3,
             'kinbote': 1,
             'raghavan': 3,
             'riker': 16,
             'prosthetic': 11,
             'sabella': 8,
             'crabby': 1,
             'bafflingly': 3,
             'subjectively': 4,
             'hound': 24,
             'compadre': 1,
             'foch': 33,
             'rationalized': 2,
             'choreographs': 1,
             'gated': 3,
             'unbelievability': 4,
             'literary': 76,
             'eggleston': 1,
             'port': 28,
             'ghosts': 181,
             'karun': 1,
             'dorset': 1,
             'gamboling': 1,
             'complimented': 13,
             'unnecessary': 307,
             'repel': 3,
             'jurisdiction': 8,
             'breakout': 14,
             'menjou': 2,
             'peugeot': 1,
             'rollercoaster': 4,
             'reworkings': 2,
             'celie': 31,
             'collude': 1,
             'inspirational': 62,
             'matthieu': 2,
             'bergstrom': 1,
             'seti': 4,
             'acin': 1,
             'unescapably': 1,
             'bedding': 5,
             'megawatt': 1,
             'feist': 2,
             'dawns': 11,
             'piranha': 5,
             'congested': 4,
             'satiricon': 1,
             'bocho': 1,
             'scandalously': 1,
             'bippy': 1,
             'saxaphone': 1,
             'gloom': 20,
             'muddies': 2,
             'throat': 122,
             'toledo': 2,
             'boning': 3,
             'erick': 1,
             'prattling': 1,
             'katryn': 1,
             'colloquial': 3,
             'beahan': 2,
             'swayzee': 4,
             'parsimonious': 1,
             'mehras': 1,
             'foremost': 49,
             'amritlal': 1,
             'browse': 2,
             'reviving': 12,
             'galaxies': 2,
             'syncer': 1,
             'snappily': 1,
             'kuno': 2,
             'resse': 1,
             'areas': 116,
             'newsweek': 5,
             'resemble': 80,
             'yound': 1,
             'dedicating': 2,
             'pfcs': 1,
             'retiring': 7,
             'bedpost': 1,
             'missourians': 1,
             'clutching': 12,
             'jumpedtheshark': 1,
             'sacks': 7,
             'horrorfilm': 1,
             'arduno': 1,
             'mendel': 5,
             'tightness': 2,
             'rauschen': 1,
             'faat': 5,
             'contini': 2,
             'deficiencies': 11,
             'nico': 4,
             'sprocket': 4,
             'synopsis': 111,
             'cupertino': 1,
             'bilborough': 1,
             'sawney': 3,
             'peddle': 6,
             'controlled': 74,
             'corrective': 1,
             'narayan': 1,
             'dood': 2,
             'until': 1776,
             'rivalling': 1,
             'squares': 14,
             'blazers': 2,
             'suitor': 13,
             'influence': 211,
             'ensor': 1,
             'humourous': 6,
             'holes': 365,
             'legion': 37,
             'factotum': 1,
             'batcave': 7,
             'daud': 1,
             'turnbull': 1,
             'buscemi': 28,
             'carachters': 1,
             'supplanted': 3,
             'reliquary': 1,
             'kimberley': 1,
             'judgemental': 5,
             'caille': 1,
             'roadblock': 7,
             'gigi': 20,
             'geographies': 1,
             'giornate': 1,
             'lowry': 5,
             'jaid': 2,
             'fjaestad': 1,
             'blots': 1,
             'always': 3239,
             'impoverished': 19,
             'karloff': 148,
             'momentary': 9,
             'gammera': 19,
             'relased': 1,
             'today': 1245,
             'ruffian': 1,
             'rebel': 103,
             'tantalising': 3,
             'labels': 13,
             'romniei': 2,
             'mclendon': 1,
             'noni': 13,
             'nenji': 1,
             'ohana': 2,
             'theological': 7,
             'dimensional': 255,
             'fichtner': 1,
             'pitchers': 1,
             'correctly': 82,
             'zebra': 2,
             'konishita': 1,
             'mstie': 4,
             'coombs': 3,
             'slapsticky': 1,
             'bucks': 124,
             'springs': 33,
             'disregarded': 10,
             'cloistered': 1,
             'vonneguts': 1,
             'noughties': 2,
             'foment': 1,
             'kln': 1,
             'minny': 1,
             'gough': 14,
             'florentine': 3,
             'coattails': 3,
             'acres': 10,
             'beeping': 3,
             'softie': 2,
             'roar': 17,
             'sigfried': 1,
             'frencified': 1,
             'cinematics': 2,
             'eds': 3,
             'ongoings': 1,
             'goldmine': 7,
             'timid': 23,
             'got': 3587,
             'websites': 14,
             'flattop': 2,
             'yetians': 2,
             'chutzpah': 6,
             'regenerating': 2,
             'karyn': 9,
             'apace': 1,
             'kneel': 3,
             'dupree': 2,
             'kundera': 11,
             'scen': 1,
             'undoubtably': 3,
             'atchison': 1,
             'questioner': 2,
             'aclear': 1,
             'cornel': 2,
             'regulations': 9,
             'pagans': 2,
             'cooly': 1,
             'garry': 11,
             'bojangles': 3,
             'seawall': 1,
             'remarried': 4,
             'worthing': 1,
             'pugilist': 4,
             'mortensen': 31,
             'elaine': 13,
             'didia': 1,
             'rekindles': 4,
             'beatings': 16,
             'truely': 9,
             'claudie': 1,
             'ingmar': 17,
             'geritol': 1,
             'stripper': 39,
             'magick': 5,
             'flashbacks': 240,
             'tarquin': 3,
             'caressing': 10,
             'walt': 63,
             'puerility': 1,
             'testimony': 24,
             'posthumously': 3,
             'backlash': 11,
             'dignities': 2,
             'magnificently': 29,
             'translators': 3,
             'voyeurs': 3,
             'lippmann': 1,
             'commonly': 22,
             'shining': 129,
             'dizzying': 19,
             'repression': 21,
             'uptake': 2,
             'busby': 44,
             'wild': 432,
             'carolina': 19,
             'monder': 1,
             'samba': 5,
             'steals': 212,
             'embarrasment': 2,
             'tinsel': 6,
             'wonderfalls': 5,
             'snatches': 9,
             'careers': 107,
             'leonidas': 1,
             'parablane': 1,
             'interfaith': 1,
             'bar': 386,
             'blowjob': 1,
             'dresch': 1,
             'slowely': 1,
             'rapist': 69,
             'newstart': 1,
             'gieldgud': 1,
             'helium': 3,
             'rydell': 3,
             'grmpfli': 1,
             'spoler': 1,
             'positions': 32,
             'inheritor': 4,
             'yup': 27,
             'mcswain': 1,
             'thunderbolt': 5,
             'manu': 11,
             'equipment': 84,
             'compensated': 22,
             'herge': 12,
             'rickman': 6,
             'targetted': 1,
             'sparklers': 1,
             'clunes': 1,
             'bloodiness': 1,
             'cordobes': 1,
             'solanki': 1,
             'prozac': 6,
             'medications': 5,
             'storage': 19,
             'incurs': 2,
             'controversies': 6,
             'pendants': 1,
             'tcp': 1,
             'bops': 9,
             'disarmed': 6,
             'norse': 5,
             'pang': 29,
             'naschy': 70,
             'shaughnessy': 1,
             'motown': 2,
             'felton': 4,
             'cascading': 2,
             'streetwalker': 3,
             'corine': 1,
             'filmable': 2,
             'soiler': 1,
             'maiga': 1,
             'dufus': 2,
             'sunnies': 2,
             'necromaniac': 2,
             'barrages': 1,
             'fiend': 43,
             'brownstone': 13,
             'subduing': 2,
             'looted': 2,
             'sealed': 32,
             'eligible': 13,
             'sailplane': 3,
             'paccino': 1,
             'resolving': 14,
             'yamaguchi': 3,
             'pyrotechnics': 13,
             'unbowed': 1,
             're': 4504,
             'tuneful': 8,
             'zirconia': 2,
             'ryck': 2,
             'roland': 26,
             'luciferian': 1,
             'dowling': 1,
             'haji': 6,
             'discomfiting': 2,
             'gluttonous': 3,
             'rhapsody': 10,
             'suggestion': 65,
             'jadoo': 1,
             'hoechlin': 18,
             'ev': 1,
             'obsessiveness': 4,
             'naya': 1,
             'busboy': 2,
             'gotham': 17,
             'baastard': 1,
             'unhappy': 95,
             'zering': 1,
             'dernier': 1,
             'bleibtreu': 4,
             'jardine': 2,
             'believer': 36,
             'wight': 1,
             'camara': 1,
             'unrequited': 9,
             'preaches': 9,
             'aborts': 3,
             'nugmanov': 1,
             'governed': 6,
             'toon': 6,
             'goodbyes': 2,
             'furlings': 1,
             'lead': 1310,
             'garza': 1,
             'layouts': 1,
             'macneille': 4,
             'preparations': 10,
             'undershorts': 1,
             'intervenes': 14,
             'uproar': 12,
             'nabokovian': 1,
             'fawn': 3,
             'incisively': 2,
             'septej': 1,
             'songmaking': 1,
             'keillor': 2,
             'menijr': 1,
             'randomness': 11,
             'grotto': 1,
             'esmond': 3,
             'libs': 7,
             'permissable': 1,
             'knack': 34,
             'budgeter': 2,
             'community': 290,
             'signoff': 1,
             'anjela': 1,
             'fcker': 2,
             'casino': 63,
             'himselfsuch': 1,
             'yaitanes': 1,
             'view': 964,
             'peddled': 2,
             'desolate': 32,
             'quicky': 1,
             'joaquim': 4,
             'obtruding': 1,
             'abductions': 1,
             'rival': 161,
             'relecting': 1,
             'keypad': 1,
             'immune': 24,
             'kampung': 2,
             'innane': 2,
             'blaster': 2,
             'magots': 1,
             'sara': 46,
             'cootie': 1,
             'worthlessness': 2,
             'bhaje': 1,
             'dirt': 76,
             'shreds': 20,
             'darnit': 1,
             'astoundingly': 16,
             'logs': 5,
             'dolby': 19,
             'harman': 4,
             'doctrinal': 2,
             'blankfield': 3,
             ...})



Let's keep the first 10000 most frequent words. As Andrew noted, most of the words in the vocabulary are rarely used so they will have little effect on our predictions. Below, we'll sort `vocab` by the count value and keep the 10000 most frequent words.


```python
vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]
print(vocab[:60])
```

    ['', 'the', '.', 'and', 'a', 'of', 'to', 'is', 'br', 'it', 'in', 'i', 'this', 'that', 's', 'was', 'as', 'for', 'with', 'movie', 'but', 'film', 'you', 'on', 't', 'not', 'he', 'are', 'his', 'have', 'be', 'one', 'all', 'at', 'they', 'by', 'an', 'who', 'so', 'from', 'like', 'there', 'her', 'or', 'just', 'about', 'out', 'if', 'has', 'what', 'some', 'good', 'can', 'more', 'she', 'when', 'very', 'up', 'time', 'no']


What's the last word in our vocabulary? We can use this to judge if 10000 is too few. If the last word is pretty common, we probably need to keep more words.


```python
print(vocab[-1], ': ', total_counts[vocab[-1]])
```

    sordid :  30


The last word in our vocabulary shows up in 30 reviews out of 25000. I think it's fair to say this is a tiny proportion of reviews. We are probably fine with this number of words.

**Note:** When you run, you may see a different word from the one shown above, but it will also have the value `30`. That's because there are many words tied for that number of counts, and the `Counter` class does not guarantee which one will be returned in the case of a tie.

Now for each review in the data, we'll make a word vector. First we need to make a mapping of word to index, pretty easy to do with a dictionary comprehension.

> **Exercise:** Create a dictionary called `word2idx` that maps each word in the vocabulary to an index. The first word in `vocab` has index `0`, the second word has index `1`, and so on.


```python
# word2idx = ## create the word-to-index dictionary here
word2idx = {}
for i,word in enumerate(vocab):
    word2idx[word] = i
word2idx
```




    {'': 0,
     'performs': 4874,
     'trailer': 1444,
     'strength': 2093,
     'uncanny': 7094,
     'gut': 5499,
     'btk': 9471,
     'stunts': 3267,
     'tail': 5467,
     'victims': 1446,
     'predict': 5665,
     'maurice': 8103,
     'crack': 4063,
     'boost': 8712,
     'protagonists': 3155,
     'wits': 9175,
     'fans': 447,
     'slick': 4515,
     'bites': 8056,
     'cage': 1779,
     'remakes': 5233,
     'year': 275,
     'senator': 7737,
     'crime': 805,
     'essentially': 2013,
     'clone': 6511,
     'sales': 7728,
     'marc': 5097,
     'group': 586,
     'attack': 1245,
     'results': 1890,
     'red': 737,
     'miniseries': 6035,
     'satirical': 5805,
     'energy': 1679,
     'often': 397,
     'fog': 5229,
     'target': 2361,
     'outdoor': 9986,
     'soup': 5540,
     'reflection': 5032,
     'penelope': 6794,
     'closer': 2396,
     'adding': 2868,
     'lifetime': 2624,
     'patients': 4656,
     'studies': 5565,
     'sometimes': 509,
     'aura': 8508,
     'aviv': 9846,
     'rogers': 2816,
     'fat': 1880,
     'imagine': 822,
     'entertainment': 701,
     'strangest': 9607,
     'ghostly': 7870,
     'wider': 6979,
     'trashy': 4303,
     'defense': 4811,
     'albert': 1960,
     'mack': 7972,
     'shaw': 4256,
     'heads': 1809,
     'unborn': 9580,
     'bonus': 4366,
     'creepy': 926,
     'villain': 984,
     'supporting': 683,
     'knights': 8167,
     'gloomy': 7345,
     'bite': 3832,
     'feet': 2162,
     'doesnt': 9228,
     'reactions': 3360,
     'mouthed': 6654,
     'thankfully': 2660,
     'vcr': 7839,
     'openly': 7931,
     'meant': 963,
     'cases': 2901,
     'mistaken': 3995,
     'females': 5166,
     'cerebral': 5987,
     'borrowed': 4892,
     'bible': 3361,
     'nothing': 164,
     'extremely': 566,
     'foley': 9821,
     'theodore': 7828,
     'massacre': 3168,
     'toole': 8363,
     'protection': 7386,
     'joins': 5200,
     'vaudeville': 8590,
     'quentin': 6383,
     'blob': 3634,
     'goof': 9455,
     'tell': 369,
     'lacks': 1486,
     'occasions': 5305,
     'cinematographic': 8690,
     'consequently': 7062,
     'fishburne': 5870,
     'remake': 1004,
     'titled': 3668,
     'tedious': 2303,
     'means': 802,
     'shepherd': 6031,
     'merrill': 9390,
     'fever': 3786,
     'andrew': 3141,
     'suggestive': 9356,
     'blockbusters': 7310,
     'juvenile': 3935,
     'encouraged': 7819,
     'graveyard': 6677,
     'newman': 5125,
     'is': 7,
     'sixty': 9432,
     'runaway': 7760,
     'bet': 2104,
     'carter': 3080,
     'sidney': 2704,
     'spoof': 2805,
     'unnerving': 8727,
     'lieutenant': 9229,
     'personal': 943,
     'evolution': 5935,
     'required': 2565,
     'laugh': 457,
     'governor': 9893,
     'enjoyment': 3098,
     'instincts': 6843,
     'contained': 3854,
     'firm': 4985,
     'sorcery': 9959,
     'wandering': 4614,
     'stock': 2027,
     'slashers': 7346,
     'walken': 3235,
     'nevertheless': 2159,
     'exhausted': 9728,
     'evelyn': 5958,
     'wolverine': 9412,
     'sant': 9440,
     '.': 2,
     'demonstrate': 6372,
     'automatically': 5326,
     'lansbury': 8586,
     'persons': 4630,
     'thomas': 2033,
     'chaos': 4108,
     'decline': 6162,
     'ernest': 7809,
     'inability': 5142,
     'pulled': 1898,
     'documents': 9347,
     'reflected': 7327,
     'comfort': 5051,
     'rugged': 9080,
     'iowa': 9847,
     'ago': 589,
     'stops': 2976,
     'robot': 2308,
     'react': 4003,
     'well': 73,
     'talents': 1935,
     'lake': 2058,
     'real': 147,
     'jeremy': 3306,
     'traveled': 8927,
     'denying': 8629,
     'possesses': 6878,
     'uncertain': 8311,
     'then': 94,
     'shore': 7084,
     'tv': 243,
     'american': 295,
     'violence': 557,
     'currie': 9721,
     'neighbor': 3278,
     'substantial': 7277,
     'existing': 7315,
     'within': 729,
     'your': 130,
     'owns': 6358,
     'novak': 4786,
     'saif': 9066,
     'products': 6950,
     'promptly': 6273,
     'realistically': 7135,
     'fare': 2368,
     'pathos': 6803,
     'benefits': 7268,
     'clerk': 6877,
     'paris': 1363,
     'destined': 5786,
     'krishna': 9579,
     'social': 1009,
     'peckinpah': 7974,
     'rivals': 7833,
     'wooden': 1618,
     'puppy': 6764,
     'speed': 2061,
     'species': 4407,
     'directed': 517,
     'principle': 6304,
     'kathryn': 5683,
     'duck': 5664,
     'diabolical': 9377,
     'anil': 6239,
     'angelina': 7322,
     'purse': 8913,
     'smallest': 9785,
     'cow': 7099,
     'throughout': 462,
     'rolled': 4938,
     'jameson': 9122,
     'circus': 5474,
     'e': 953,
     'grotesque': 5211,
     'animators': 8953,
     'implausible': 4009,
     'seemingly': 1551,
     'colleagues': 6261,
     'hostage': 7294,
     'expected': 854,
     'afterwards': 3498,
     'placed': 2583,
     'raft': 9512,
     'dunst': 7841,
     'weapon': 3125,
     'europeans': 6478,
     'canceled': 6680,
     'role': 214,
     'warrant': 8005,
     'enjoyed': 500,
     'chance': 565,
     'nonsense': 1825,
     'berkeley': 6395,
     'terribly': 1893,
     'sentiments': 9811,
     'invites': 5306,
     'establishment': 6605,
     'keith': 4708,
     'arranged': 6678,
     'quest': 2652,
     'finest': 1867,
     'tcm': 6904,
     'unlike': 1003,
     'achieving': 9641,
     'circle': 4165,
     'stepped': 9387,
     'token': 5853,
     'sports': 2222,
     'wal': 9638,
     'foil': 7074,
     'racing': 5865,
     'raised': 2812,
     'painting': 3327,
     'winchester': 5779,
     'charlie': 1262,
     'shoe': 6756,
     'clueless': 5663,
     'chicks': 4657,
     'steer': 6972,
     'harron': 7597,
     'reunion': 3079,
     'bullets': 3579,
     'overacts': 9246,
     'pig': 4196,
     'from': 39,
     'frances': 7087,
     'cuts': 1891,
     'coverage': 8818,
     'robbie': 9904,
     'beetle': 8092,
     'radical': 6047,
     'adults': 1449,
     'afford': 4131,
     'farmer': 5468,
     'effect': 934,
     'scarlet': 7249,
     'selection': 5469,
     'quarters': 9612,
     'askey': 9185,
     'jeep': 9737,
     'handling': 5307,
     'hypnotic': 8461,
     'verhoeven': 3710,
     'washing': 9812,
     'about': 45,
     'latino': 8286,
     'trite': 3072,
     'presumably': 3490,
     'difficulty': 5580,
     'israeli': 6796,
     'method': 4177,
     'victim': 1326,
     'stallone': 4696,
     'argue': 3896,
     'accounts': 6476,
     'involvement': 3841,
     'snowman': 7039,
     'infinitely': 7798,
     'austen': 4735,
     'rejects': 7019,
     'ya': 4147,
     'natali': 7203,
     'hug': 8912,
     'champion': 5656,
     'darth': 6894,
     'viewing': 812,
     'frankenstein': 4540,
     'terrorism': 8238,
     'mst': 2692,
     'amber': 9977,
     'allegory': 9581,
     'eerie': 3238,
     'has': 48,
     'myrtle': 7194,
     'irs': 9783,
     'terrorist': 4160,
     'brenda': 4870,
     'pan': 4148,
     'progresses': 4605,
     'france': 2172,
     'dumber': 6621,
     'effectively': 2609,
     'distraction': 6703,
     'ian': 3398,
     'adaption': 6592,
     'neither': 1066,
     'mines': 8355,
     'isolation': 6377,
     'equals': 9469,
     'farm': 3807,
     'kris': 6961,
     'surround': 7762,
     'desperately': 2709,
     'cuban': 6704,
     'olds': 6782,
     'filth': 5507,
     'gone': 809,
     'waving': 9287,
     'grand': 1720,
     'groove': 8414,
     'offering': 3960,
     'julie': 2773,
     'injury': 7185,
     'jazz': 3975,
     'barely': 1178,
     'val': 7759,
     'mafia': 3411,
     'wishes': 3035,
     'patrick': 2269,
     'highlight': 2443,
     'pervert': 8179,
     'died': 1109,
     'stiff': 3392,
     'cliffhanger': 6548,
     'uniformly': 5883,
     'discovered': 1941,
     'glimpses': 7221,
     'mild': 3368,
     'sellers': 4305,
     'boss': 1325,
     'inclined': 7820,
     'theories': 6593,
     'competently': 9534,
     'rap': 3564,
     'wash': 6883,
     'game': 485,
     'paxton': 5372,
     'point': 211,
     'west': 1188,
     'masks': 5129,
     'centers': 4585,
     'instead': 304,
     'williams': 1567,
     'weeks': 2434,
     'pm': 7783,
     'pretends': 7982,
     'bimbo': 8104,
     'motives': 4178,
     'truly': 366,
     'allowing': 3479,
     'questionable': 4589,
     'clips': 2922,
     'chorus': 4287,
     'sin': 2998,
     'offs': 5641,
     'saves': 3205,
     'undertones': 9537,
     'empty': 1886,
     'penn': 5246,
     'twins': 5044,
     'ensemble': 3105,
     'denouement': 7083,
     'durbin': 8964,
     'stevenson': 8618,
     'duh': 7148,
     'cigarette': 6854,
     'strange': 666,
     'silver': 3281,
     'produce': 2216,
     'burgess': 8200,
     'baldwin': 5081,
     'working': 763,
     'banks': 8143,
     'stands': 1388,
     'questions': 1181,
     'rendering': 7598,
     'psychiatric': 9151,
     'les': 4927,
     'sf': 6423,
     'perhaps': 376,
     'showcase': 4638,
     'sweden': 7075,
     'fate': 1928,
     'edmund': 7821,
     'are': 27,
     'robertson': 6498,
     'glowing': 5442,
     'tragic': 1550,
     'slightly': 1057,
     'naive': 2304,
     'carmen': 5988,
     'zatoichi': 9457,
     'hope': 439,
     'trying': 265,
     'wish': 640,
     'entrance': 7180,
     'categories': 8156,
     'demonic': 5957,
     'projection': 9862,
     'vengeful': 9372,
     'comic': 681,
     'pay': 973,
     'demands': 3855,
     'bitchy': 8373,
     'remade': 5605,
     'hayes': 7122,
     'moody': 4337,
     'accomplishment': 8531,
     'warhols': 8247,
     'dunne': 8404,
     'hinted': 7329,
     'quantum': 6501,
     'accurately': 5897,
     'text': 2985,
     'nostalgic': 4387,
     'weirdness': 8123,
     'morals': 6594,
     'din': 4976,
     'connie': 7981,
     'stumble': 7641,
     'compelled': 4751,
     'butcher': 6179,
     'empire': 3575,
     'hooper': 6705,
     'bodyguard': 8046,
     'visceral': 8698,
     'jail': 2795,
     'cassie': 8051,
     'thank': 1258,
     'debbie': 6690,
     'resolved': 6262,
     'bow': 5435,
     'drunk': 1795,
     'rich': 998,
     'says': 550,
     'advised': 6468,
     'temper': 7738,
     'ourselves': 3115,
     'chen': 8314,
     'age': 544,
     'ideas': 990,
     'directions': 5040,
     'parodies': 8093,
     'thelma': 6638,
     'elizabeth': 2744,
     'suggest': 1445,
     'le': 3393,
     'johnny': 1696,
     'others': 402,
     'disgusting': 2282,
     'mythology': 6776,
     'unfortunately': 465,
     'sleeve': 9212,
     'by': 35,
     'dominick': 8920,
     'useful': 4462,
     'dare': 3291,
     'alive': 1209,
     'parent': 3687,
     'sheen': 6826,
     'totally': 478,
     'undercover': 8394,
     'revolt': 6512,
     'imitating': 8893,
     'hand': 496,
     'oscar': 709,
     'monologues': 8446,
     'the': 1,
     'sees': 1067,
     'mate': 3614,
     'disdain': 9199,
     'sirk': 4522,
     'fluid': 6821,
     'sequences': 829,
     'ties': 4113,
     'terrorists': 4679,
     'bars': 5890,
     'mindless': 3037,
     'inform': 8304,
     'related': 2436,
     'stalwart': 9987,
     'smoothly': 9167,
     'dukakis': 8608,
     'mockery': 8836,
     'tradition': 2918,
     'loyal': 4260,
     'beth': 8329,
     'support': 1406,
     'driving': 1926,
     'laurie': 6319,
     'canada': 3275,
     'sailors': 9083,
     'search': 1769,
     'minnelli': 6412,
     'talent': 659,
     'spies': 5911,
     'invasion': 4592,
     'wives': 4102,
     'vets': 9328,
     'hudson': 3237,
     'corruption': 4264,
     'foot': 2001,
     'poo': 9169,
     'carla': 4842,
     'experts': 8053,
     'steady': 5531,
     'collar': 7374,
     'entertain': 2799,
     'carefully': 3362,
     'holmes': 2903,
     'fool': 2293,
     'sweeping': 8729,
     'sean': 1969,
     'warnings': 9925,
     'bumps': 9960,
     'gimmick': 6123,
     'readily': 6891,
     'mankind': 4684,
     'challenging': 4718,
     'subject': 852,
     'bout': 7851,
     'unattractive': 6254,
     'names': 1411,
     'barnes': 8088,
     'garden': 3980,
     'photos': 4440,
     'spirits': 4103,
     'destroying': 4499,
     'boyer': 5824,
     'hallmark': 7744,
     'tone': 1133,
     'exception': 1377,
     'muscular': 8908,
     'obnoxious': 2895,
     'background': 959,
     'producer': 1270,
     'thailand': 7399,
     'vividly': 6216,
     'immediately': 1215,
     'feminine': 6712,
     'phase': 6830,
     'lions': 8392,
     'accident': 1675,
     'gerard': 4566,
     'fascinated': 4719,
     'turkish': 7404,
     'melinda': 8487,
     'citizen': 3534,
     'progression': 8029,
     'smoke': 3706,
     'voices': 2301,
     'in': 10,
     'cocky': 9195,
     'cheaper': 9234,
     'society': 876,
     'sweat': 7149,
     'capturing': 4736,
     'knock': 3276,
     'discussions': 9298,
     'miraculously': 7795,
     'together': 292,
     'freedom': 2116,
     'reporter': 2338,
     'overblown': 6606,
     'anime': 2126,
     'courtroom': 6747,
     'seemed': 461,
     'importantly': 3487,
     'ritual': 6726,
     'proceed': 7789,
     'scoop': 5677,
     'viewpoint': 6903,
     'singer': 1897,
     'reluctantly': 7587,
     'greater': 2775,
     'complete': 587,
     'achieved': 3270,
     'flowers': 5679,
     'masterpiece': 969,
     'items': 5201,
     'depiction': 2796,
     'unexpectedly': 5128,
     'fades': 7784,
     'sections': 8026,
     'fetish': 7673,
     'horrendous': 3357,
     'lesbian': 2413,
     'parrot': 7204,
     'notice': 1475,
     'sandy': 8109,
     'stopped': 2208,
     'cell': 2732,
     'choosing': 5965,
     'brooke': 7181,
     'happiness': 2614,
     'teeth': 2684,
     'gods': 5918,
     'burned': 3829,
     'exploited': 6713,
     'offended': 4146,
     'reaching': 4422,
     'school': 383,
     'mode': 5352,
     'allows': 2039,
     'lung': 9288,
     'flames': 7032,
     'fan': 332,
     'taped': 6595,
     'underdeveloped': 7873,
     'phrases': 9495,
     'genie': 5729,
     'alley': 5812,
     'eliminate': 9423,
     'visions': 5548,
     'cambodia': 9600,
     'bunch': 745,
     'facial': 2722,
     'performers': 3091,
     'exploit': 6429,
     'crass': 8986,
     'deniro': 5014,
     'significance': 5074,
     'bury': 9230,
     'van': 1141,
     'flimsy': 6413,
     'cover': 1085,
     'unfolds': 4120,
     'calm': 4802,
     'latest': 2450,
     'adore': 6424,
     'financial': 4088,
     'cynicism': 8713,
     'yourself': 612,
     'pretty': 184,
     'bsg': 5975,
     'specifically': 4239,
     'leia': 9289,
     'hanging': 2324,
     'historians': 9263,
     'manipulate': 8794,
     'especially': 258,
     'maturity': 8591,
     'debut': 1984,
     'scandal': 8607,
     'lauren': 6082,
     'phenomenon': 5634,
     'israel': 4849,
     'mister': 7860,
     'mute': 6801,
     'fisher': 3455,
     'doug': 7413,
     'another': 160,
     'startling': 5996,
     'expressions': 3014,
     'sensitivity': 8124,
     'sinks': 8125,
     'few': 171,
     'leno': 8934,
     'flamenco': 9639,
     'appeals': 6414,
     'documentary': 645,
     'everybody': 1339,
     'harmless': 5502,
     'opponents': 9064,
     'jed': 9034,
     'insanity': 5938,
     'whole': 222,
     'trapped': 2595,
     'manner': 1345,
     'witches': 4558,
     'fassbinder': 6543,
     'guide': 3530,
     'saga': 4223,
     'jagger': 7370,
     'whenever': 1924,
     'factor': 2299,
     'explores': 5599,
     'tones': 7548,
     'classy': 6430,
     'mix': 1472,
     'looking': 264,
     'buffy': 7867,
     'language': 1074,
     'torch': 8825,
     'under': 459,
     'sullavan': 9738,
     'disturbed': 3986,
     'kamal': 8997,
     'whilst': 1847,
     'bitch': 5414,
     'racial': 4309,
     'audiences': 1185,
     'great': 86,
     'power': 651,
     'mining': 9482,
     'paula': 5981,
     'hunter': 2136,
     'bottle': 4606,
     'ripoff': 7508,
     'delicious': 6240,
     'spoofs': 8169,
     'credited': 5295,
     'werewolf': 1913,
     'sexuality': 3094,
     'volume': 6452,
     'solo': 4254,
     'resemblance': 4032,
     'winston': 9342,
     'scientific': 3697,
     'obtain': 5942,
     'vintage': 6469,
     'alcohol': 4914,
     'dr': 856,
     'earl': 4759,
     'margaret': 3823,
     'pour': 8967,
     'sack': 7724,
     'hiring': 9611,
     'capacity': 7256,
     'morgana': 9045,
     'offer': 1442,
     'admission': 7792,
     'budding': 8177,
     'justified': 5929,
     'structured': 8233,
     'bleed': 7232,
     'assuming': 5666,
     'council': 9876,
     'tank': 5067,
     'ang': 8072,
     'ludicrous': 2737,
     'protest': 7136,
     'next': 371,
     'melody': 5662,
     'victorian': 6315,
     'noticeable': 6415,
     'sterling': 7132,
     'tim': 1710,
     'walt': 5978,
     'outsider': 9786,
     'matt': 2253,
     'vanilla': 6552,
     'definitive': 6208,
     'ny': 6993,
     'ninety': 7474,
     'illogical': 4290,
     'curly': 5275,
     'willem': 9601,
     'recognize': 2503,
     'aunts': 9814,
     'problem': 436,
     'mcintire': 9188,
     'fathers': 5113,
     'forever': 1407,
     'alliance': 7155,
     'clay': 6444,
     'conclusion': 1160,
     'sassy': 5481,
     'charges': 7662,
     'wander': 6217,
     'schneider': 6646,
     'melodrama': 2579,
     'resident': 4819,
     'versus': 3950,
     'orchestral': 9757,
     'angie': 6655,
     'optimism': 8632,
     'mental': 1711,
     'gift': 3412,
     'trouble': 1092,
     'explosions': 4008,
     'fanning': 8488,
     'happen': 581,
     'weekly': 7397,
     'affecting': 7776,
     'awareness': 7245,
     'meantime': 6860,
     'buried': 3514,
     'chose': 2444,
     'organs': 9357,
     'coma': 7093,
     'impressions': 8416,
     'published': 6258,
     'wrote': 1016,
     'sans': 9048,
     'culkin': 8209,
     'mustache': 8027,
     'hoped': 3187,
     'pseudo': 3903,
     'launch': 7241,
     'october': 7195,
     'gritty': 2493,
     'chemical': 8546,
     'grinch': 4161,
     'socialist': 9894,
     'lucio': 9408,
     'captured': 1980,
     'lists': 8574,
     'torture': 1756,
     'standout': 6191,
     'ninjas': 8020,
     'lovable': 3188,
     'bombs': 5831,
     'week': 1222,
     'dawn': 3334,
     'faded': 7785,
     'lastly': 7205,
     'carl': 3981,
     'frankie': 4619,
     'jamie': 3463,
     'attendant': 8225,
     'uplifting': 5033,
     'screw': 5727,
     'bridges': 5767,
     'described': 2183,
     'diving': 8004,
     'pointing': 7105,
     'changed': 1170,
     'cheese': 2977,
     'fitting': 3313,
     'breast': 6994,
     'awful': 368,
     'scale': 2367,
     'endearing': 3283,
     'delivers': 1523,
     'criticisms': 7858,
     'opens': 1995,
     'assembled': 6339,
     'maintained': 9969,
     'priests': 8924,
     'page': 1412,
     'dub': 4413,
     'russia': 5114,
     'laputa': 6825,
     'nutshell': 7387,
     'fanatic': 6964,
     'delivered': 2113,
     'character': 105,
     'clip': 4954,
     'pecker': 7515,
     'bust': 7524,
     'spoilers': 1011,
     'blake': 3932,
     'stress': 4314,
     'hood': 2914,
     'department': 2522,
     'score': 591,
     'pale': 6307,
     'cox': 3399,
     'resulting': 5015,
     'judy': 4267,
     'grisly': 8811,
     'heroes': 1674,
     'frontal': 5989,
     'herrings': 8674,
     'slapped': 6774,
     'hk': 5380,
     'cute': 1007,
     'cole': 3426,
     'bounty': 6537,
     'swept': 5823,
     'architect': 8931,
     'dwarf': 8251,
     'pertwee': 8417,
     'property': 4768,
     'mutants': 8592,
     'ted': 2483,
     'comical': 2810,
     'kay': 4624,
     'realises': 7498,
     'someday': 5647,
     'idiot': 2632,
     'biker': 6952,
     'anchors': 8418,
     'calling': 2735,
     'other': 82,
     'abstract': 8489,
     'add': 751,
     'containing': 5939,
     'hill': 2110,
     'vegas': 4431,
     'mayor': 4414,
     'satisfaction': 6856,
     'epitome': 7948,
     'eleanor': 9539,
     'presumed': 9518,
     'suspicion': 7064,
     'left': 314,
     'duties': 9321,
     'inevitable': 3417,
     'beverly': 5296,
     'commitment': 6702,
     'cedric': 7555,
     'cannon': 5240,
     'lionel': 6725,
     'contestant': 7578,
     'get': 76,
     'taught': 4425,
     'introduces': 4296,
     'suspects': 3009,
     'jarring': 6253,
     'invited': 5489,
     'crucial': 4812,
     'bat': 2844,
     'filmmakers': 1025,
     'inept': 2750,
     'motel': 7579,
     'massive': 2530,
     'honestly': 1228,
     'suit': 1701,
     'toy': 2821,
     'greg': 4970,
     'notch': 2473,
     'usa': 2889,
     'pitt': 2515,
     'variations': 9583,
     'essential': 2919,
     'bleak': 3692,
     'over': 120,
     'showtime': 9565,
     'billion': 8721,
     'crew': 1021,
     'stepmother': 8048,
     'muddled': 5294,
     'therapy': 7133,
     'drawing': 3830,
     'enjoys': 3897,
     'shamelessly': 9014,
     'did': 121,
     'interviewed': 7447,
     'annoyed': 3200,
     'feeling': 538,
     'than': 74,
     'fantasy': 912,
     'boom': 4511,
     'tendency': 6563,
     'strikes': 3343,
     'lived': 1433,
     'femme': 5058,
     'voiced': 4317,
     'verge': 7727,
     'organization': 7393,
     'model': 2119,
     'eats': 5720,
     'control': 1115,
     'routine': 2453,
     'groundbreaking': 8515,
     'local': 702,
     'referring': 6263,
     'remarkable': 1712,
     'gamera': 6520,
     'kentucky': 9409,
     'shark': 3251,
     'rewarding': 6459,
     'mood': 1282,
     'serial': 1402,
     'experiment': 2813,
     'bikini': 6525,
     'voight': 3913,
     'rider': 6425,
     'apocalypse': 7269,
     'yells': 8287,
     'commando': 9815,
     ...}



### Text to vector function

Now we can write a function that converts a some text to a word vector. The function will take a string of words as input and return a vector with the words counted up. Here's the general algorithm to do this:

* Initialize the word vector with [np.zeros](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html), it should be the length of the vocabulary.
* Split the input string of text into a list of words with `.split(' ')`. Again, if you call `.split()` instead, you'll get slightly different results than what we show here.
* For each word in that list, increment the element in the index associated with that word, which you get from `word2idx`.

**Note:** Since all words aren't in the `vocab` dictionary, you'll get a key error if you run into one of those words. You can use the `.get` method of the `word2idx` dictionary to specify a default returned value when you make a key error. For example, `word2idx.get(word, None)` returns `None` if `word` doesn't exist in the dictionary.


```python
def text_to_vector(text):
    word_vector = np.zeros(len(vocab), dtype=np.int_)
    #word_vector *0 = 0
    for word in text.split(" "):
        if word in vocab:
            word_vector[word2idx[word]] += 1
        #else:
    return np.array(word_vector)        
    
    pass
```

def text_to_vector(text):
    word_vector = np.zeros(len(vocab),dtype=np.int_)
    for word in text.split(" "):
        idx = word2idx.get(word,None)
        if idx == None:
            continue
        else:
            word_vector[word2idx[word]] += 1
    return np.array(word_vector)
        
        

If you do this right, the following code should return

```
text_to_vector('The tea is for a party to celebrate '
               'the movie so she has no time for a cake')[:65]
                   
array([0, 1, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
```       


```python
text_to_vector('The tea is for a party to celebrate '
               'the movie so she has no time for a cake')[:65]
```




    array([0, 1, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])



Now, run through our entire review data set and convert each review to a word vector.


```python
word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0])
```


```python
# Printing out the first 5 word vectors
word_vectors[:5, :23]
```




    array([[ 18,   9,  27,   1,   4,   4,   6,   4,   0,   2,   2,   5,   0,
              4,   1,   0,   2,   0,   0,   0,   0,   0,   0],
           [  5,   4,   8,   1,   7,   3,   1,   2,   0,   4,   0,   0,   0,
              1,   2,   0,   0,   1,   3,   0,   0,   0,   1],
           [ 78,  24,  12,   4,  17,   5,  20,   2,   8,   8,   2,   1,   1,
              2,   8,   0,   5,   5,   4,   0,   2,   1,   4],
           [167,  53,  23,   0,  22,  23,  13,  14,   8,  10,   8,  12,   9,
              4,  11,   2,  11,   5,  11,   0,   5,   3,   0],
           [ 19,  10,  11,   4,   6,   2,   2,   5,   0,   1,   2,   3,   1,
              0,   0,   0,   3,   1,   0,   1,   0,   0,   0]])



### Train, Validation, Test sets

Now that we have the word_vectors, we're ready to split our data into train, validation, and test sets. Remember that we train on the train data, use the validation data to set the hyperparameters, and at the very end measure the network performance on the test data. Here we're using the function `to_categorical` from TFLearn to reshape the target data so that we'll have two output units and can classify with a softmax activation function. We actually won't be creating the validation set here, TFLearn will do that for us later.


```python
Y = (labels=='positive').astype(np.int_)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]
trainX, trainY = word_vectors[train_split,:], to_categorical(Y.values[train_split], 2)
testX, testY = word_vectors[test_split,:], to_categorical(Y.values[test_split], 2)
```


```python
trainY
```

## Building the network

[TFLearn](http://tflearn.org/) lets you build the network by [defining the layers](http://tflearn.org/layers/core/). 

### Input layer

For the input layer, you just need to tell it how many units you have. For example, 

```
net = tflearn.input_data([None, 100])
```

would create a network with 100 input units. The first element in the list, `None` in this case, sets the batch size. Setting it to `None` here leaves it at the default batch size.

The number of inputs to your network needs to match the size of your data. For this example, we're using 10000 element long vectors to encode our input data, so we need 10000 input units.


### Adding layers

To add new hidden layers, you use 

```
net = tflearn.fully_connected(net, n_units, activation='ReLU')
```

This adds a fully connected layer where every unit in the previous layer is connected to every unit in this layer. The first argument `net` is the network you created in the `tflearn.input_data` call. It's telling the network to use the output of the previous layer as the input to this layer. You can set the number of units in the layer with `n_units`, and set the activation function with the `activation` keyword. You can keep adding layers to your network by repeated calling `net = tflearn.fully_connected(net, n_units)`.

### Output layer

The last layer you add is used as the output layer. Therefore, you need to set the number of units to match the target data. In this case we are predicting two classes, positive or negative sentiment. You also need to set the activation function so it's appropriate for your model. Again, we're trying to predict if some input data belongs to one of two classes, so we should use softmax.

```
net = tflearn.fully_connected(net, 2, activation='softmax')
```

### Training
To set how you train the network, use 

```
net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
```

Again, this is passing in the network you've been building. The keywords: 

* `optimizer` sets the training method, here stochastic gradient descent
* `learning_rate` is the learning rate
* `loss` determines how the network error is calculated. In this example, with the categorical cross-entropy.

Finally you put all this together to create the model with `tflearn.DNN(net)`. So it ends up looking something like 

```
net = tflearn.input_data([None, 10])                          # Input
net = tflearn.fully_connected(net, 5, activation='ReLU')      # Hidden
net = tflearn.fully_connected(net, 2, activation='softmax')   # Output
net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
model = tflearn.DNN(net)
```

> **Exercise:** Below in the `build_model()` function, you'll put together the network using TFLearn. You get to choose how many layers to use, how many hidden units, etc.


```python
# Network building
def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    
    #### Your code ####
    
    net = tflearn.input_data([None,10000])
    net = tflearn.fully_connected(net, 200, activation='ReLU') 
    net = tflearn.fully_connected(net, 25, activation='ReLU') 
    net = tflearn.fully_connected(net, 2, activation='softmax')
    
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
    #### Your code ####
    
    model = tflearn.DNN(net)
    return model
```

## Intializing the model

Next we need to call the `build_model()` function to actually build the model. In my solution I haven't included any arguments to the function, but you can add arguments so you can change parameters in the model if you want.

> **Note:** You might get a bunch of warnings here. TFLearn uses a lot of deprecated code in TensorFlow. Hopefully it gets updated to the new TensorFlow version soon.


```python
model = build_model()
```

## Training the network

Now that we've constructed the network, saved as the variable `model`, we can fit it to the data. Here we use the `model.fit` method. You pass in the training features `trainX` and the training targets `trainY`. Below I set `validation_set=0.1` which reserves 10% of the data set as the validation set. You can also set the batch size and number of epochs with the `batch_size` and `n_epoch` keywords, respectively. Below is the code to fit our the network to our word vectors.

You can rerun `model.fit` to train the network further if you think you can increase the validation accuracy. Remember, all hyperparameter adjustments must be done using the validation set. **Only use the test set after you're completely done training the network.**


```python
# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=10)
```

## Testing

After you're satisified with your hyperparameters, you can run the network on the test set to measure its performance. Remember, *only do this after finalizing the hyperparameters*.


```python
predictions = (np.array(model.predict(testX))[:,0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:,0], axis=0)
print("Test accuracy: ", test_accuracy)
```

    Test accuracy:  0.4392


## Try out your own text!


```python
# Helper function that uses your model to predict sentiment
def test_sentence(sentence):
    positive_prob = model.predict([text_to_vector(sentence.lower())])[0][1]
    print('Sentence: {}'.format(sentence))
    print('P(positive) = {:.3f} :'.format(positive_prob), 
          'Positive' if positive_prob > 0.5 else 'Negative')
```


```python
sentence = "Moonlight is by far the best movie of 2016."
test_sentence(sentence)

sentence = "It's amazing anyone could be talented enough to make something this spectacularly awful"
test_sentence(sentence)
```

    Sentence: Moonlight is by far the best movie of 2016.
    P(positive) = 0.500 : Positive
    Sentence: It's amazing anyone could be talented enough to make something this spectacularly awful
    P(positive) = 0.500 : Positive



```python

```
