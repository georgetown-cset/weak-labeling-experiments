<center><h1>Snorkel For Programmatic Data Labeling</h1></center>

## Synopsis & Project Motivation
In recent years, Machine Learning has slowly made its way into almost every industry. From Marketing to Agriculture and everything in between. As machine leaning models increase in complexity, so is the need for labelled training data. However, obtaining labelled training data still remains a huge bottle-neck for any machine learning application. The [Snorkel Project](https://snrokel.org) developed by a Stanford University lab and backed by Google attempts to solve this problem by using [weak supervision](https://www.snorkel.org/blog/weak-supervision).     

This project attempts to use Snorkel to create weak labels for large amounts of training data at CSET using weak supervision. The data is primarily paper abstracts, which we attempt to explore their relevance to various [WIPO Application Areas](https://wipo.org)

## Project Structure
This project is split into two phases. Phase 1 of the project explores creating weak labels for paper abstracts in the Driver Detection WIPO application area. This phase of the project also involves a set of evaluations performed throughout the project, as well as hand labelled data used for the evaluations. PHase 2 of the project involves scaling the project to cover all other WIPO application areas (54 total). It also involves scaling the training data and the labeling functions. This phase of the project involves a different method of model evaluation since there were no gold labels to run the traditional evaluations.   

The [documentation directory](documentation/), contains weekly documentation for experiments tried that week and weekly progress of the project. Phase 1 runs from week 1 to about week 6, while phase 2 runs from week 6 to week 10. The data directory contains sample datasets used throughout the project. 


## Project Implementation & Experiments
At its core, Snorkel uses labelling functions to create weak labels for datasets. A labelling function is a simple rule-based function that labels a data point. These functions are often written by a domain expert in the field of study, and they could range from simple heuristic functions like a function that checks the word length of a data point to more complex functions that can employ an external knowledge-base or a third party model. The output of labelling functions are used to create a probabilistic or generative model, which is then creates the weak labels for each data point, along with a confidence value for each label. 

In phase 1 of this project, we attempt to label papers as either relevant or irrelevant to the Driver Detection application area. It is worth knowing that labeling functions either label a data point or abstain (no label). 
> Note: A data point is the paper title and abstract concatenated together. 
Below is a sample labeling function that checks a data point and marks it as RELEVANT if the data point has the words "driver detection"
```python
from snorkel.labeling import labeling_function
data_point = "paper abstract..."

@labeling_function
def lf_driver_detection(data_point):
    return RELEVANT if "driver detection" in data_point else ABSTAIN
```

## Running the Project
> This project requires Python 3

1. Clone the repositiory     
   ```shell
   $ git clone https://github.com/georgetown-cset/weak-labeling-experiments.git
   $ cd weak-labeling-experiments
   ```
2. Install project dependencies    
    ```shell
    $ pip3 install -r requirements.txt
    ```
3. Change directories into the weak labeling directory
   ```shell
   $ cd weak_labeling
   ```
4. Build a Label Model for a single application area using the application area id   
   ```shell
   $ python lf_generator.py --wipo_id id_0005 --label_out "../output/labels_for_id_0005.jsonl" --save_model "../output/model_id_0005.pkl"
    ```
5. Run Unittest
    ```shell
    $ python -m unittest discover -s tests -p "test_*.py"
    ```

## Examples
Build a label model for the driver detection application area with evaluations by using gold labels
```shell
$ cd weak-labeling-experiments/weak_labeling
$ python weak_labels.py --data ../data/training_data.csv
```

Build label models for a particular WIPO application area using the application area ID. In this case, our    
the application area id is `id_0005`
```shell
$ cd weak-labeling-experiments/weak_labeling
$ python lf_generator.py --wipo_id id_0005 \
    --label_out "../output/wipo_logs/labels_for_id_0005.jsonl" \
    --save_model "../output/model_export/model_id_0005.pkl"
```

## Conclusion
This project was an experimental project through which we explored the potential use cases and limitations of Snorkel. There are plans to continue exploring weak-labeling as a tool for CSET projects in the future, and this project would serve as an initial pilot to those future ventures.

## Contributors
Intern: [Collins Nji](https://github.com/collinsnji)    
Project Mentor: [Jennifer Melot](https://github.com/jmelot)    

