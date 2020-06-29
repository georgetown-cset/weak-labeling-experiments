# Week #3 Documentation of tasks
### Collins Nji

### Week Outline  
__Monday__ - Scaled training dataset and code clean up     
__Tuesday__ - Added more LFs to differentiate AVs from driver/vehicle recognition     
__Wednesday__ - Improvements to model accuracy     
__Thursday__ -  Model tweaking and false positive analysis      
__Friday__ -  Juneteenth (Georgetown University Holiday) 🎉      


### Overview
This week I worked on tweaking the labeling functions from last week to improve accuracy. I scaled the training dataset     
from about 200 data points to about 7000 data points. (See `$PROJECT_ROOT/weak_labeling/data_import.py`). With 200 data points, both the     
`MajorityLabelVoter` and the `LabelModel` produced an accuracy score of about 68%. The `MajorityLabelVoter` takes the     
majority vote on a per-data point basis: if more LFs voted RELEVANT than IRRELEVANT, label it RELEVANT (and vice versa).    
The `LabelModel` produces a single set of noise-aware training labels, that is, confidence-weighted labels which can be     
used to train a classifier. I calculated the accuracy score as follows: 

```python
majority_acc = majority_model.score(
  L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
label_model_acc = label_model.score(
  L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")

# Outputs:
# Majority Vote Accuracy:   68.2%
# Label Model Accuracy:     68.2%
```

***Note 1:*** It is worth noting that the above code uses a *`random` tie breaker*, which randomly choose among tied option      
using a deterministic hash. Later on I changed the tie breaker to `abstain`     
>*Note to self: Learn more about the tie breaker policy, and why abstaining improves the model score*

I continued tweaking the models and learning more from the changes I made. One thing I did was filter out duplicated   
data points between the training data and the test data. There were about 15 duplicated data points. This did not have   
any significant impact on the model accuracy score  

I did some code clean up and wrote more LFs to better highlight IRRELEVANT data points (See lines 67-91 in week-3/weak_labels.py)    
I also added some LFs at an attempt differentiate AVs from driver/vehicle recognition. Adding more LFs improved the model scores    
by about 1.5 percentage points. 

```diff
+ majority_acc = majority_model.score(
-    L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
+    L=L_test, Y=Y_test, tie_break_policy="abstain", metrics=["f1", "accuracy"])
+label_model_acc = label_model.score(
-    L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
+    L=L_test, Y=Y_test, tie_break_policy="abstain", metrics=["f1", "accuracy"])
```

The changes above resulted in the following model improvements to the model accuracy
```text
Majority Vote Accuracy:   90.0%
Majority Vote F1 Score:   94.3%

Label Model Accuracy:     86.8%
Label Model F1 Score:     92.5%
```

I then proceeded to calculate how many documents abstained during the training phase of the model. I'm not exactly sure what the    
impact of the abstained documents have on the overall performance yet, but I plan to look into this some time next week.    
Lastly, I did some preliminary inquires into false positives, false negatives etc. And Below are some of the results I could get

```text
ABSTAIN data points: 2017 


true positives: 62
false positives: None
false negatives: 10
true_negatives: 4


abstained positives 1
abstained negatives 106
```

## Notes to self
- I think the `abstained negatives` value is high due to the LFs targeting very specific keywords
- `f1 score (f1_micro in sckitlearn)` calculate metrics globally by counting the total true positives, false negatives and false positives.
  
### Summary of Task list
- [x] ~~Write summary for results~~   
- [x] ~~Understand what F1 Score is and how it relates to the accuracy of the LFs I wrote~~      
      ~~(true positives, false positives, true negatives etc)~~
- [x] ~~Write documentation of everything I did this week~~
- [ ] Write unit tests for data import function and keyword LF definition    
- [ ] Measure precise time it takes to fit the LabelModel on the test data
- [ ] Measure if class imbalance in gold set affects model accuracy 
  
### Other Random Things I tried
- Tried calculating an MCC (Matthew Correlation Coefficient) using SciKit Learn (could not exactly figure that one out)
- Adding a big query config object to the data import function to avoid typing the same query over and over


### Things I may need help understanding
1. All of James messages in thread (Mostly the acronyms?)
2. Why I get the warning `Metrics calculated over data points with non-abstain labels only`    
   when I use the `abstain` method as a tie breaker when calculating the model score
