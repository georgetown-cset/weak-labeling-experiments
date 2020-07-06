# Week #5 Documentation of tasks
### Collins Nji

### Week Outline  
__Monday__ - Code review with Jennifer and and overview of other WIPO application areas     
__Tuesday__ - Data preparation and initial LFs     
__Wednesday__ - Read lots of documentation on LFs and `fit`ting a model (class_balance)        
__Thursday__ -  Testing LF performance      
__Friday__ -  Refinements to code/refactoring + documentation      

## Overview
This week, I worked on expanding the labelling functions to cover other WIPO AI Application categories, including agriculture, finance, etc. I started the week with some preliminary data preparations, then proceeded to writing some LFs using the new keywords and categories.       


I created a simple csv file (keyword_categories.csv) with four columns `Id, Category, Subcategory, LikelyPhrases`.
```csv
Id,Category,Subcategory,LikelyPhrases
id_0000,Agriculture,N/A,"agriculture, farming, cultivation, animal breeding, agronomy, pesticide, pest control, agrochemical, fertilizer"
id_0001,Arts and humanities,N/A,"fine arts, performing arts, architecture, language translation, media arts, music, cinema, cinematography, movie direction, writing, painting, sculpting, photography, theatre"
id_0002,Banking and finance,N/A,"banking, finance, insurance, reinsurance, insurable, trading, liability"
id_0003,Business,General,"electronic, commerce, enterprise, computing, customer, service, digital cash, e-commerce infrastructure, electronic data interchange, electronic funds transfer, online shopping, online banking, secure online transactions, online auctions, forecasting, marketing, video content discovery, recruitment, enterprise information systems, intranets, extranets, enterprise resource planning, enterprise applications, data centers, business process management, business process modeling, business process management systems, business process monitoring, cross-organizational business processes, business intelligence, enterprise architectures, enterprise architecture management, enterprise architecture frameworks, enterprise architecture modeling, service-oriented architectures, event-driven architectures, business rules, enterprise modeling, enterprise, ontologies, taxonomies, vocabularies, enterprise data management, business-it alignment, it architectures, it governance, enterprise computing infrastructures, enterprise interoperability, enterprise application integration, information integration, information interoperability"
id_0004,Business,Customer services,customer service
```
> _Note: I added the ID category so that I could easily reference individual categories and create LFs without causing a conflict in names_

I also parsed this file and transformed it to a JSON file. This was not immediately used, but I think I may have use for it later on.       

I wrote a few LFs to begin with (unfortunately, I deleted the outputs before documenting). However, they basically work similar or the same to the LFs I later generated. Since I had multiple LFs to work with, I wanted to come up with a way of programmatically generating them.

## The Process
The first method I used in generating the LFs involved creating a unique list of all the keywords across all the WIPO categories. Each word representing an LF. This method generated a total of ~570 LFs which I applied to the training data from last week. It was clear that some LFs preformed better than others when looking at the output of 
```python
lf_analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
```
> _Note: Since I could not build a `LabelModel` to get a score, I used coverage information from the above analysis to indicate performance. The higher the better_

From this, I noticed some stop words like `"a" (adverb), "it" (abbv. of Information Technology)` had a really high coverage. So I manually removed these words from the LikelyPhrases column of the keywords dataframe. This step was helpful in identifying words/phrases that could be relevant or too broad for LFs. It is also at this step that I noticed some words had spelling errors that may have lead to unexpected results. Instead of manually correcting every incorrect spelling, I chose to use the python `TextBlob` module to do so programmatically. 

_It is worth noting that this step takes about ~__45 seconds__ to complete the spell check on all the words in the `LikelyPhrases` column. (timed with `tqdm`) Which is a lot slower than I expected it to be. I may have to end up manually correcting the spelling errors on that column if necessary_ 

After cleaning up some of the likely phrases and removing stop words, I decided to try another approach with generating the LFs. Instead of generating a single LF for every unique word in the likely phrases column of the dataframe, I decided to generate LFs using the category ID as the LF, and then passing the `likely phrases` as keywords to the LF. This resulted in 54 LFs. Below is the output of the analysis while using this method:  
```python
                j Polarity  Coverage  Overlaps  Conflicts
    id_0048  48       []  0.000000  0.000000        0.0
    id_0004   4      [1]  0.000144  0.000144        0.0
    id_0031  31      [1]  0.000718  0.000718        0.0
    id_0000   0      [1]  0.002155  0.002155        0.0
    id_0040  40      [1]  0.002873  0.002873        0.0
    id_0024  24      [1]  0.004884  0.004884        0.0
    id_0007   7      [1]  0.006607  0.006607        0.0
    id_0050  50      [1]  0.007325  0.007325        0.0
    id_0047  47      [1]  0.007325  0.007325        0.0
    id_0039  39      [1]  0.007325  0.007182        0.0
    id_0002   2      [1]  0.010055  0.010055        0.0
    id_0036  36      [1]  0.010629  0.010629        0.0
    id_0027  27      [1]  0.010773  0.010629        0.0
    id_0029  29      [1]  0.011204  0.011204        0.0
    id_0023  23      [1]  0.012353  0.012209        0.0
    id_0046  46      [1]  0.013358  0.013358        0.0
    id_0045  45      [1]  0.024849  0.024849        0.0
    id_0037  37      [1]  0.037489  0.037489        0.0
    id_0001   1      [1]  0.059897  0.059897        0.0
    id_0008   8      [1]  0.065786  0.065786        0.0
    id_0005   5      [1]  0.079144  0.079144        0.0
    id_0035  35      [1]  0.082448  0.082017        0.0
    id_0018  18      [1]  0.090348  0.089486        0.0
    id_0017  17      [1]  0.102844  0.101982        0.0
    id_0042  42      [1]  0.119650  0.119650        0.0
    id_0003   3      [1]  0.123815  0.123815        0.0
    id_0013  13      [1]  0.156133  0.156133        0.0
    id_0014  14      [1]  0.163172  0.162741        0.0
    id_0015  15      [1]  0.167337  0.167337        0.0
    id_0022  22      [1]  0.179259  0.179115        0.0
    id_0011  11      [1]  0.190032  0.189601        0.0
    id_0012  12      [1]  0.193910  0.193910        0.0
    id_0020  20      [1]  0.198937  0.198506        0.0
    id_0010  10      [1]  0.213588  0.213444        0.0
    id_0043  43      [1]  0.216174  0.216030        0.0
    id_0009   9      [1]  0.260126  0.260126        0.0
    id_0033  33      [1]  0.265154  0.265154        0.0
    id_0044  44      [1]  0.291296  0.291296        0.0
    id_0028  28      [1]  0.309394  0.309394        0.0
    id_0030  30      [1]  0.311548  0.311548        0.0
    id_0016  16      [1]  0.314134  0.314134        0.0
    id_0053  53      [1]  0.428182  0.427894        0.0
    id_0025  25      [1]  0.474433  0.474145        0.0
    id_0021  21      [1]  0.510198  0.509049        0.0
    id_0049  49      [1]  0.583453  0.583453        0.0
    id_0019  19      [1]  0.607440  0.606866        0.0
    id_0051  51      [1]  0.641913  0.641339        0.0
    id_0006   6      [1]  0.683281  0.681988        0.0
    id_0026  26      [1]  0.727664  0.727664        0.0
    id_0038  38      [1]  0.756535  0.756392        0.0
    id_0041  41      [1]  0.759121  0.758977        0.0
    id_0052  52      [1]  0.780379  0.780236        0.0
    id_0032  32      [1]  0.793450  0.793450        0.0
    id_0034  34      [1]  0.800632  0.800632        0.0
```
As expected, `id_0052` (Driver detection/recognition) performs a lot better than most other LFs because the training data is slightly tailored to suit that LF. From these results I was able to go back and see which LFs performed the least (performance here is a measure of coverage where the higher the better). After looking at which LFs performed the list, I noticed it tends to be WIPO fields with the least number of likely terms that performed worse. 

## Notes to Self and what I learned
> __Highlight__: I'm officially halfway through interning at CSET! That's crazy 🎉😅

- `class_balance` in the `fit`ting step is calculated using the ground truth labels stored in `Y_dev`, if the `Y_dev` param is specified. Otherwise, it assumes a uniform class distribution. That is,
    ```python
    class_balance = ((1/cardinality) * np.ones(cardinality))

    # Output: [0.5 0.5] for a cardinality of 2
    # where cardinality is the number of classes
    ``` 
- Don't make every keyword an individual LF. It's not very informative
- It takes ~ 1 min 35 seconds to apply all 54 LFs (could you improve this?)
- Document your code early enough!!! 😡
  
## Summary of task list
- [x] Parsed wipo categories to a csv file, then produced a json file from output (I may use this later)
- [x] Added an ID column to the csv to easily reference the different categories
- [x] Try generating LFs for every keyword in the categories (eg `keyword_ld_entertainment`)
- [x] Use the `TextBlob` module in Python to correct spelling errors
- [x] Try generating keywords using the `Id` as LF name, and the `likely phrases` column as a keyword list.
- [x] Write documentation of task
- [ ] Code clean up