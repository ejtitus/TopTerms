TopTermsInLabeledData README, Eric Titus, October 2, 2015

This code works on the brand labled datasets to extract terms of importance to each brand using a TFIDF weighting scheme.
The top terms are saved to an external text file, and wordcloud plots are generated for each brand.

The data should be in a csv with the headings ['Date','Handle','Source','Content','Brand']

This code relys on the following external packages:

-numpy
-scipy
-pandas      "pip install pandas" at the command line will install (on linux)
-nltk         natural language toolkit-need the most recent version "pip install -U nltk"
-matplotlib   "pip install matplotlib" used for plotting
-seaborn      "pip install seaborn" Seaborn makes the plots look better
-scikit-learn "pip install scikit-learn" This is the package that does the TFIDF weighting
-wordcloud    "pip install wordcloud" This package turns the top terms into wordclouds
