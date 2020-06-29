# Week #4 Documentation of tasks
### Collins Nji

### Week Outline  
__Monday__ - Code review with Jennifer and next steps     
__Tuesday__ - Precision/Recall analysis     
__Wednesday__ - Unit tests and effects of class imbalance     
__Thursday__ -  More testing      
__Friday__ -  Some more refactoring      


## Overview
This week I worked on code refactoring and detailing analysis. I added a precision and recall score calculation, refactored code to align with typical python standards, and lastly, added unit tests for some functions.     

I refactored the code in `weak_labels.py` so that all the code is in an `if __name=="__main"` block, and added command line parsers. I believe my command line arguments could use some improvements, but for now they work as expected.    

I used metric functions from the sklearn library to calculate the precision and recall scores.

```python
p_score = precision_score(y_true=Y_test, y_pred=Y_pred, average='weighted')
r_score = recall_score(y_true=Y_test, y_pred=Y_pred, average='weighted', labels=np.unique(Y_pred))

# Output
# Label Model Accuracy:     86.84%
# Label Model F1 Score:     92.54%
# Precision Score:          95.22%
# Recall Score:             36.07%
```
>_Note: It is worth noting that I used the `'weighted'` average here which calculates metrics for each label, and find their average weighted by support. This method alters 'macro' to account for label imbalance_

>__Note to Self__: Investigate what effects the other types of averages have on the precision/recall scores and the overall model accuracy  

### Notes to self
- Dude, why is the recall score so low? Is it supposed to be low?
- After reading more about the recall score, I tired adding a `class_balance=` parameter to the training step:
  ```python
    model = LabelModel(cardinality=2, verbose=True)
    model.fit(L_train=L_train, n_epochs=800, log_freq=100, class_balance=[0.673, 0.327])
  ```
  This resulted in a much higher recall score, and an improvement in the Label Model accuracy. But I'm not exactly sure why that was the case. Below is the output
    ```
    Label Model Accuracy:     88.52%
    Label Model F1 Score:     81.08%
    Precision Score:          89.11%
    Recall Score:             88.52%
    ```
### Summary of tasks list
- [x] Refactor weak_labels.py so all code is either inside an `if __name__ == "__main__"` block, or inside a function
- [x] Write code to pickle your LabelModel
- [x] Write a test for `keyword_lookup` function using the `unittest` library
- [x] Use the `time` library to print out how long each major stage takes 
- [x] Use the `argparse` library to allow the user to pass in an input file and an output file. For the output file, write out your metrics to a text file named (input `filename+"_run_"+datetime.txt`) instead of printing them

### Other random things/thoughts
- The LF_applier step is what takes the most time (~4.4 seconds). I used the time module to get this value, but this step produces a progress bar when you run it, which also shows the time it took to ran the step (hence using the time module may not be that necessary)
- Wow! I totally underestimate how hard it is to write unittest for two lines of code! This proved to be the hardest thing I had to do all week!