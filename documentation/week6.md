# Week #6 Documentation of tasks
### Collins Nji

### Week Outline  
__Monday__ - Code review with Jennifer        
__Tuesday__ - Refining keyword categories         
__Wednesday__ - Functions for new data imports        
__Thursday__ -  Coding + testing      
__Friday__ -  Performance reviews     
__Saturday__ - More code and testing      

## Overview
This week, I worked on refining the Labeling Functions for the WIPO application categories. I reimplemented various methods for handling the LFs, including generating individual LFs for each category instead of generating a massive list of unrelated LFs (more on this below). I also worked on reimplementing the data import functions and finally, added a basic implementation of multiprocessing (which I believe will certainly need a code review)

## The Process
Usually when the week starts, I try to put my thoughts together before jumping into coding. This week was no different, except, it was a bit of a struggle understanding some of the tasks I had to do. (Thanks to Jennifer for resolving some of them)  

> Some important points to note: 
> 1. The dataset I used for testing the LFs was very small (~530 rows) which may possibly affect performance
> 2. I did not perform any code clean-up on my code (commented out code here and there)
> 3. The model fitting step using `multiprocessing` clearly needs some work
> 4. This commit does not contain output files from `lf_analysis_summary`. I zipped the output and it is available on Google Drive


This week, I re-wrote how each category handles its labelling functions. Instead of creating a `N x 1` list of labelling functions which lumps all the keywords together, I created a dictionary using the category IDs as the key, with the category LFs in a list as the value. Hence,
```
{
    "id_0000": [ Lf1, Lf2, ..., Lfn ],
       .
       .
       .
    "id_xxxx": [ Lf1, Lf2, ..., Lfn ]
}
```
While this method was a lot easier to work with, it also presented a problem: how do I pass a `label` to the individual labeling functions? Essentially, how do I mark an LF as relevant or irrelevant. To address this problem, I used the words in the `LikelyPhrases` column from the WIPO Application table which did not appear in the `ImprovedLikelyPhrases` column as negative (irrelevant) labels. The reason I used this approach is because some of the words on the `LikelyPhrases` column may contextually be related to the WIPO application area, but may be too broad to provided any significant analysis.

```python
wordlist = keyword["ImprovedLikelyPhrases"].split(sep=", ")
negative_wordlist = list(set(keyword["LikelyPhrases"].split(sep=", ")) - set(wordlist))
positive_keyword_lfs = [make_keyword_lf(keywords=[wl], label=RELEVANT) for wl in wordlist]
negative_keyword_lfs = [make_keyword_lf(keywords=[wl], label=IRRELEVANT) for wl in negative_wordlist]
```
> _see `generate_lfs()` in [../weak_labelling/lf_generator.py](../weak_labeling/lf_generator.py#L61)_

It is also worth pointing out that I rewrote parts of the `keyword_lookup` function to use regular expressions instead of full word searches, which improved the coverage of certain LFs. Before implementing this, many LFs had a `0.00` coverage.
```python
if (len(word.split(sep=" ")) > 2):
    word_regex = word.strip().replace(" ", ".*")
    return label if re.search(fr"(?i){word_regex}", current_data_point) else ABSTAIN
elif any(word in current_data_point for word in keywords):
    return label
else:
    return ABSTAIN
```
> _Note to self: Measure the impact of using a regular expression_ 

Lastly, It is worth pointing out that I did not pickle the label models in the code, but I can write a quick patch to do so when necessary.  


## Task list
- [x] Review [commit](https://github.com/georgetown-cset/weak-labeling-experiments/commit/7a395b5db87eddef65afd2a809a99cb3311e98a8) to get labels
- [x] Create new "train set" from MAG using ~1000 docs (or fewer) from each MAG FieldOfStudy. Compare predicted relevant coverage for each WIPO category’s LabelModel vs each FieldOfStudy
- [x] Try making recognition LF more specific or removing it
- [x] Write a less expensive query for pulling training data from Big Query
- [x] Use `multiprocessing` library for training label models
- [x] Compare "performance" (see below) of only positive LFs generated from WIPO key phrases vs positive + all irrelevant WIPO key phrase LFs as negative
- [ ] Have a function in this script that takes a WIPO category as a parameter and that returns a list of LabelingFunctions (almost complete)


## What I need help with
- Analysis (I'm having a hard time understanding how to measure accuracies, especially in relation to the `FieldsOfStudy` [see Google Doc])
- Code review (A code review session will be nice. I believe there are some parts of my code that could be improved)
- Evaluating the various LFs generated (related to the first bullet point)
- Do I need to train the `LabelModels` on a new bigger dataset?
- Multiprocessing (phew! figuring this one out is tough)

## Random Thoughts
- Currently the `make_keyword_lf` function test if any of the keywords passed into the function is available in the title or abstract of the paper. What if all the words passed in the keyword param match? That could be used as an indicator to weight the LF a bit more?
- I still can't think of a great way of estimating the `class balance` param
- Document your code and also remove all the unused lines of code. It's confusing
- Find better coding music. Breaking Benjamin is too sad 😭