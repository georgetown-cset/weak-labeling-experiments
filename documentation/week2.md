# Week #2 Documentation of Tasks
### Collins Nji

### Overview  
__Monday__ - Reading Snorkel documentation  
__Tuesday__ - Getting started with Snorkel and the Spam Tutorial*  
__Wednesday__ - More documentation reading/coding*  
__Thursday__ -  Coding (wrote LFs)
__Friday__ -  Improving LFs and documenting tasks for week #2

<sub>[*] <i>I had a few doctor's appointments this week that set me back by a few hours</i></sub>

### Overview
This week I mostly worked on understanding the basics of Snorkel and its application in weak labeling. I started by reading the Snorkel documentation pages and the spam classification tutorial. After reading some of the documentation, I started some basic implementation using the data from last week (see `week1-data`).    

After looking at the data again, I realized I missed a few relevancy tags on the data. (See commit [9631432](https://github.com/georgetown-cset/weak-labeling-experiments/commit/96314320985ed3b60b5861ceabf71ae8e3778740) for more details). I quickly added the relevancy tag to these data points then proceeded to writing the initial labeling functions. 

### Problem Approach
- I split the dataset into a training and testing set using the method below. This splits the dataset with 20% of the data as a testing set, and the other 80% for training. 
```python
# some lines have been intentionally left out. See week-2/labeling_fn.py for full code
df = pd.read_csv('merged_data.csv', prefix=None).sample(frac=1, random_state=120).reset_index(drop=True)
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=120, stratify=df.relevancy)
```
- Next, I filled in missing abstracts with the string `"NA"` so that I could easily work with the data. I'm  positive there could be a better way of doing this, but I'll stick with this one for now
- I proceeded to write the LFs, testing for coverage as I went on

### Some decisions/assumptions made in writing the LFs
* I ignored the sentiment preprocessor since most abstracts don't really have a direct sentiment
* I avoided using plurals of keywords in keyword lookup LFs. I'm not exactly sure how this would have impacted the coverage, but I do plan on testing this at some point. I think a potential solution to this will be to lemmatize the keywords used in keyword LFs
* I assumed all data points with a missing abstract or an abstract written in any language other than English was irrelevant. I understand that some documents could have relevant keywords regardless of the language the abstract is written in, so these particular LFs will need further review.
  

### Some plans/ideas for next week (week #3)

- I plan to use some form of lemmatization (from `TextBlob` package) to find more relevant keywords   
- I plan to read/explore more information about the Snorkel preprocessors (`spaCy` and `nlp_labeling_function`)    
- __Write a detail interpretation of summary results__     