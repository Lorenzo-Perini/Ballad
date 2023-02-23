# BALLAD

`BALLAD` (Budget allocation for Active Learning and Learning to reject in Anomaly Detection) is a GitHub repository containing the **BALLAD** [1] algorithm.
It refers to the paper titled *How to Allocate your Label Budget? Choosing between Active Learning and Learning to Reject in Anomaly Detection*, presented at the 1st AAAI Workshop on Uncertainty Reasoning and Quantification in Decision Making on February 13th 2023 in Washington D.C.

Read the ArXiv version here: [[pdf](https://arxiv.org/pdf/2301.02909.pdf)].

## Abstract

Anomaly detection attempts at finding examples that deviate from the expected behaviour. Usually, anomaly detection is tackled from an unsupervised perspective because anomalous labels are rare and difficult to acquire. However, the lack of labels makes the anomaly detector have high uncertainty in some regions, which usually results in poor predictive performance or low user trust in the predictions. 
One can reduce such uncertainty by collecting specific labels using Active Learning (AL), which targets examples close to the detector’s decision boundary. Alternatively, one can increase the user trust by allowing the detector to abstain from making highly uncertain predictions, which is called Learning to Reject (LR). One way to do this is by thresholding the detector’s uncertainty based on where its performance is low, which requires labels to be evaluated. Although both AL and LR need labels, they work with different types of labels: AL seeks strategic labels, which are evidently biased, while LR requires i.i.d. labels to evaluate the detector’s performance and set the rejection threshold. Because one usually has a unique label budget, deciding how to optimally allocate it is challenging. 
In this paper, we propose a mixed strategy that, given a budget of labels, decides in multiple rounds whether to use the budget to collect AL labels or LR labels. The strategy is based on a reward function that measures the expected gain when allocating the budget to either side. We evaluate our strategy on 18 benchmark datasets and compare it to some baselines.

## Contents and usage

The repository contains:
- BALLAD.py, a set of functions that allow to run our method BALLAD to smartly allocate your label budget;
- Notebook.ipynb, a notebook showing how to use BALLAD on an artificial 2D dataset;

To use BALLAD, import the github repository or simply download the files. You can also find the benchmark datasets at this [[link](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/)]. Please check the dataset's names inside the function "load_dataset" in BALLAD.py to replicate our experiments.


## Budget allocation for Active Learning and Learning to reject in Anomaly Detection (BALLAD)

Given a dataset with attributes **X**, we use SSDO [[paper](https://ieeexplore.ieee.org/document/8594877)] as a semi-supervised anomaly detector to learn from partially labeled examples. We simulate k allocation rounds. We initialize the problem by (1) training SSDO with no labels and setting a default rejection threshold to 0.1, and (2) collecting random labels for LR and for AL in the first two allocation rounds. This allows us to compute the initial rewards by **measuring how the detector varies from (1) to (2)**: for LR, we measure the variation after re-setting the validation threshold; for AL, we measure the variation after re-training the detector with the new labels. Then, we start the allocation loop. In each round, we allocate the budget to the option (LR or AL) with the **highest reward**, and we update the reward using the new labels. We propose two alternative reward functions: 1) the **entropy reward** looks at the detector’s probabilities, either for prediction (AL), or for rejection (LR); 2) the **cosine reward** considers the predicted class labels, either anomaly yes/no (AL), or rejection yes/no (LR).

Given a dataset **X** with true labels **y**, we first split it into training/validation/test using the proportion 40/40/20. You can download the benchmark datasets and specify their path and their name (or set them as empty strings to use our 2D toy dataset):
```python
from BALLAD import load_dataset
X_train, y_train, X_val, y_val, X_test, y_test, contamination = load_dataset(dataset_name, data_path, random_state = 331)
```

Then, our algorithm is applied as follows:

```python
from BALLAD import *
run_BALLAD(X_train, y_train, X_val, y_val, X_test, y_test, contamination, metric, tot_budget, pool_budget, c_r, c_fp, c_fn, seed, plots)

# contamination = proportion of anomalies in the training set [[see here](https://arxiv.org/pdf/2210.10487.pdf)], metric = 'cosine' or 'entropy'
# tot_budget is the integer number of total labels, while pool_budget is the integer number of labels per allocation round
# c_r, c_fp, c_fn are the costs for rejection, false positives and false negatives
# seed allows replicability, and plots is a boolean value to get the heat-map of the model in each allocation round
```

## Dependencies

The `run_BALLAD` function requires the following python packages to be used:
- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)
- [Scipy](http://www.scipy.org)
- [Pandas](https://pandas.pydata.org/)
- [Anomatools](https://github.com/Vincent-Vercruyssen/anomatools)


## Contact

Contact the author of the paper: [lorenzo.perini@kuleuven.be](mailto:lorenzo.perini@kuleuven.be).


## References

[1] Perini, L., Giannuzzi, D. and Davis, J., 2023: *How to Allocate your Label Budget? Choosing between Active Learning and Learning to Reject in Anomaly Detection*, arXiv preprint arXiv:2301.02909.