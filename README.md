# **Loan Default Risk Prediction using XGBoost**

## **Table of Contents**
 - [Purpose](#purpose)
 - [Reference Solution](#reference-solution)
 - [Reference Implementation](#reference-implementation)
 - [Optimizing the E2E Reference Solution with Extension for scikit-learn](#optimizing-the-e2e-reference-solution-with-extension-for-scikit-learn)

## Purpose

US lenders issue trillions of dollars in new and refinanced mortgages every year, bringing the total mortgage debt to very high levels year after year. At the same time, mortgage delinquencies (i.e. non-payment of loan installments) usually represent a significant percentage, representing a huge debt risk to the bearer. In order for a Financial Organization to meet its desired risk profile, it is pivotal to build a good understanding of the chance that a particular debt may result in a delinquency.

Organizations are increasingly relying on powerful AI models to gain this understanding and are using that to build powerful tools for predictive analysis. However, these models do not come without their own set of complexities. With expanding and/or changing data, these models must be assessed periodically for their suitability towards current customers, and updated to accurately capture the current environment in a timely manner when needed.

Furthermore, as loan prediction systems are highly impactful from a societal point of view, it is no longer enough to build models that only make accurate predictions. Fair predictions are required to build an ethical AI, which could go a long way for an organization to build trust in their AI systems.

## Reference Solution

In this reference kit, we provide an example for training and utilizing an AI model using XGBoost to predict the probability of a loan default from client characteristics and the type of loan obligation. We will also provide a brief introduction to a few tools that can be used for an organization to analyze the fairness/bias that may be present in each of their trained models. These can be saved for audit purposes as well as to study and adjust the model for the sensitive decisions that this application must make. Finally, we will show how to speed up loan default predictions from an XGBoost model using Intel® optimized tools.

## Key Implementation Details

The reference kit implementation is a reference solution to the described use case that includes:

  1. A reference E2E architecture to arrive at an AI solution with an [XGBoost](https://xgboost.readthedocs.io) classifier whose input data comes from a [scikit-learn](https://scikit-learn.org) pipeline
  2. An Optimized reference E2E architecture enhanced with Intel® optimizations for XGBoost through [scikit-learn-intelex](https://uxlfoundation.github.io/scikit-learn-intelex)

## Reference Implementation

### Expected Input-Output

**Input**                                 | **Output** |
| :---: | :---: |
| Client Features         | Predicted probability between [0,1] for client to default on a loan |

**Example Input**                                 | **Example Output** |
| :---: | :---: |
| ***ID***, ***Attribute 1***, ***Attribute 2*** <br> 1, 10, "X" <br> 2, 10, "Y" <br> 3, 2, "Y" <br> 4, 1, "X" | [{'id' : 1 , 'prob' : 0.2}, {'id" : 2', 'prob' : 0.5}, {'id' : 3, 'prob' : 0.8}, {'id' : 4, 'prob' : 0.1} |9


### Dataset

The dataset used for this demo is a set of 32581 simulated loans, which can be founder under the following link:
https://www.kaggle.com/datasets/laotse/credit-risk-dataset

It has 11 features including customer and loan characteristics and one response which is the final outcome of the loan.

**Feature** | **Description** |
| :---: | :---: |
| person_age | Age of client |
| person_income | Income of client |
| person_home_ownership | Whether the client owns a home |
| person_emp_length | Length of the clients employment in years |
| loan_intent | The purpose of the loan issued |
| loan_grade | The grade of the loan issued |
| loan_amnt | The amount of the loan issued |
| loan_int_rate | The interest rate of the loan issued |
| loan_percent_income | Percent income |
| cb_person_default_on_file | Whether the client has defaulted before |
| cb_person_cred_hist_length | The length of the clients credit history |
| **loan_status** | Whether this loan ended in a default (1) or not (0)

For demonstrative purposes we make one important modification to the original dataset before experimentation using the the [`data/prepare_data.py`](data/prepare_data.py) script.

* Adding a synthetic bias_variable
    
    For the purpose of demonstrating fairness in an ML model later, we will add a bias value for each loan default prediction. This value will be generated randomly using a simple binary probability distribution as follows:
    ```
    If the loan is defaulted i.e. prediction class 1:
      assign bias_variable = 0 or 1 with the probability of 0 being 0.65

    if the loan is not defaulted i.e. prediction class 0:
      assign bias_variable = 0 or 1 with the probability of 0 being 0.35
    ```
    **Feature** | **Description** |
    | :---: | :---: |
    | bias_variable | synthetic biased variable |

    For fairness quantification, we will define that this variable should belong to a [protected class](https://en.wikipedia.org/wiki/Fairness_(machine_learning)) and `bias_variable = 1` is the privileged group.

    This variable is NOT used to train the model as the expectation is that it should not be used to make decisions for fairness purposes.

Finally, the original dataset is splitted into two parts - 70% of the observations for training, and 30% for holdout testing, on which the quality of the fitted model from only the training data will be evaluated.

### Model Training

The first step to build a default risk prediction system is to train an ML model. In this reference kit, we choose to use an XGBoost classifier on the task of using the features for a client and loan, and outputting the probability that the loan will end in a default. This can then be used downstream when analyzing whether a particular client will default across many different loan structures in order to reduce and analyze risk to the organization. XGBoost classifiers have been proven to provide excellent performance when dealing with similar predictive tasks such as fraud detection and predictive health analytics.  

#### Data Pre-Processing

Before passing the data into the model, we transform a few of the features in the dataset using a [scikit-learn pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to obtain better performance. The XGBoost library is compatible with the scikit-learn framework, and model objects from this library can be seamlessly embedded into the feature pre-processing pipeline as a final step.

The pre-processing that we will perform in detail is:

#### Categorical Features to One-Hot Encodings

Features `person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file` are all transformed to use a One-Hot Encoding to be fed into the XGBoost classifier. To avoid overfitting, only the most common categories will be used here. Note that, as of XGBoost version 2, categorical features can also be used as such without one-hot encoding, but when the cardinality of features is low, one-hot might still be a better option.

#### Imputation of Missing Values

Features `loan_int_rate`, `person_emp_length`, `cb_person_cred_hist_length` are all imputed using the median value to fill in any missing values that may be present in collection of the dataset.

### Fairness Evaluation

In many situations, accuracy is not the only consideration for deploying a model to production. For certain sensitive applications, it is also necessary to verify and quantify to what degree a model may be using information to make biased predictions, which may amplify certain inequities. This study can broadly be defined as understanding the bias and fairness of a machine learning model, which is an [actively developing field of research in ML](https://arxiv.org/pdf/1908.09635.pdf).  

To accommodate this challenge, in this reference kit, we will demonstrate the computation of a few metrics to quantify the fairness of predictions, focusing on parity between the privileged and the non-privileged groups in our previously introduced `bias_variable`. Briefly, under parity constraints, the computed metrics should be independent of the protected variable, largely performing the same whether measured on the privileged or non-privileged subgroups. A more through discussion of parity measures for fairness can be found in the link above as well as [here](https://afraenkel.github.io/fairness-book/content/05-parity-measures.html).

Computationally, after a model is trained or updated, we will report the following *ratios predictive metrics for the privileged and non-privileged groups* on a hold out test set

- **positive predictive value (PPV)**
- **false discovery rate (FDR)**
- **negative predictive value (NPV)**
- **false omission rate (FOR)**
- **true positive rate (TPR)**
- **false negative rate (FNR)**
- **true negative rate (TNR)**
- **false positive rate (FPR)**

As described above, under parity considerations, for these metrics to be independent of the protected variable, the ratio of these values should be around 1.0. Significant deviations above or below 1.0 may indicate bias that needs to be further investigated.

### Model Serving

The saved model from each model iteration can be used on new data with the same features to infer/predict the probability of a loan default. This can be deployed in a number of ways. For simplicity, in this example we will apply the model on a large batch of data loaded from a parquet file, but in practice one might want to use microservices for serving models.

### Software Requirements and Setup

To run this reference kit, a Python installation with the libraries specified in [requirements.txt](requirements.txt) is needed. A conda environment with these requirements is also specified in file [environment.yaml](environment.yaml) for ease of installation.

To run this reference kit, first clone this git repository, which can be done by executing the follow command in a terminal:

```shell
git clone https://www.github.com/oneapi-src/loan-default-risk-prediction
cd loan-default-risk-prediction
```

To set up the required Python environment with the necessary dependencies, it is recommended to create a virtual environment and then install them with `pip` - for example, assuming a Linux environment:

```shell
python -m venv loan_default
source loan_default/bin/activate
pip install requirements.txt
```

Alternatively, the environment can be set up through conda ([miniforge](https://github.com/conda-forge/miniforge) distribution is highly recommended) as follows:

```shell
conda env create --file=environment.yaml
conda activate loan_default
```

### Reference Implementation

In this section, we describe the process of building the reference solution using the scripts that we have provided, assuming that these scripts are being executed from the root of the git repository that was cloned earlier.

### Downloading the Data

The data for this example can be downloaded from this [Kaggle* link](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) in different ways. As a first step, you may or may not need to register an account if you do not have one already.

Then, download the data from the 'Download' button at the top-left of this page, and place it under the root folder of the repository containing this example:
https://www.kaggle.com/datasets/laotse/credit-risk-dataset

Alternatively, the data can be downloaded from a terminal after [setting up a Kaggle* API configuration](https://github.com/Kaggle/kaggle-api), for example by creating a file `~/.kaggle/kaggle.json` containing the necessary credentials. Assuming a Linux* environment where the credentials are set up, the following command should download the data:

```shell
curl -L -o credit-risk-dataset.zip \
    https://www.kaggle.com/api/v1/datasets/download/laotse/credit-risk-dataset
```

After downloading the data, it is necessary to extract the file it contains into the `data/` folder, which can be done as follows:
```shell
unzip credit-risk-dataset.zip -d data/
```

### Data Preparation

The script `prepare_data.py` will add the bias variable and split the data into a training and a test set. This is required for the following step.

To execute this script with default arguments, run:

```shell
python data/prepare_data.py
```

Documentation for this script:
```
usage: prepare_data.py [-h] [--bias_prob BIAS_PROB]

options:
  -h, --help            show this help message and exit
  --bias_prob BIAS_PROB
                        probability bias_variable=0 if loan is defaulted. probability
                        bias_variable=1 if loan is not defaulted.
```

### Model Building Process

The `run_training.py` script *reads the data*, *trains a feature preprocessor*, *trains an XGBoost Classifier* (the last two in a combined pipeline), and finally *saves the model* (serialized the Python objects), which can then be used for future inference.

To execute the script with default arguments, run:

```shell
python src/run_training.py
```

The script takes the following arguments:

```
usage: run_training.py [-h] [--save_model_path SAVE_MODEL_PATH] [--train_file TRAIN_FILE]
                       [--test_file TEST_FILE] [--logfile LOGFILE]

options:
  -h, --help            show this help message and exit
  --save_model_path SAVE_MODEL_PATH
                        Path to save the fitted model. If not provided, does not save.
  --train_file TRAIN_FILE
                        Data file for training (parquet format).
  --test_file TEST_FILE
                        Data file for testing (parquets format).
  --logfile LOGFILE     Log file to output benchmarking results to.
```

The output of this script is a saved model `saved_models/classifier_model.pkl`. In addition, the fairness metrics on a holdout test will also be shown as below (these are subject to changes over library versions):

```
INFO:root:Model fitting time : 0.173551 seconds
INFO:root:              precision    recall  f1-score   support

           0       1.00      0.90      0.95     19744
           1       0.60      0.97      0.75      3063

    accuracy                           0.91     22807
   macro avg       0.80      0.94      0.85     22807
weighted avg       0.94      0.91      0.92     22807

INFO:root:AUROC : 0.913240
Parity Ratios (Privileged/Non-Privileged):
        PPV : 0.96
        FDR : 3.59
        NPV : 1.13
        FOMR : 0.35
        TPR : 0.98
        FNR : 1.03
        TNR : 1.00
        FPR : 1.09

```

For the `bias_variable` generative process described above, we can see that certain values strongly deviate from 1, indicating that the model may have detected some bias and does not seem to be making equitable predictions between the two groups.  

In comparison, we can adjust the generative process so that the `bias_variable` is explicitly fair independent of the outcome:

```

If the loan is defaulted i.e. prediction class 1:
  assign bias_variable = 0 or 1 with the probability of 0 being 0.5

if the loan is not defaulted i.e. prediction class 0:
  assign bias_variable = 0 or 1 with the probability of 0 being 0.5
  
```

(to do so: `python data/prepare_data.py --bias_prob=0.5`)

and the resulting fairness metrics when re-executing `run_training.py` will be:

```
Parity Ratios (Privileged/Non-Privileged):
        PPV : 1.00
        FDR : 1.00
        NPV : 0.99
        FOMR : 1.08
        TPR : 0.98
        FNR : 1.03
        TNR : 1.00
        FPR : 1.03
```
indicating that the model is not biased along this protected variable.

A thorough investigation of fairness and mitigation of bias is a complex process that *may require multiple iterations of training and retraining the model*, potentially excluding some variables, reweighting samples, and investigation into sources of potential sampling bias. A few further resources on fairness for ML models, as well as techniques for mitigation include [this guide](https://afraenkel.github.io/fairness-book/intro.html) and [the `shap` package](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Explaining%20quantitative%20measures%20of%20fairness.html).

### Running Inference

To use this model to make predictions about probabilities of loan defaults on new data, we can use the `run_inference.py` script which takes in a saved model and a dataset to predict on.

To execute this script with the default arguments, run:

```shell
python src/run_inference.py
```

The script takes the following arguments:

```
usage: run_inference.py [-h] [--intel] [--silent] [--saved_model SAVED_MODEL]
                        [--input_file INPUT_FILE] [--logfile LOGFILE]

options:
  -h, --help            show this help message and exit
  --intel               Toggle to use Intel-optimized prediction routines
  --silent              Don't print predictions.
  --saved_model SAVED_MODEL
                        Saved model file from 'run_training.py'.
  --input_file INPUT_FILE
                        Input file for inference
  --logfile LOGFILE     Log file to output benchmarking results to.
```

## Optimizing the E2E Reference Solution with Extension for scikit-learn

After fitting a credit default model, new prospective loans will be assessed through it as clients requests them, perhaps in real time if offered through online services. This typically requires calculating default probabilities very quickly.

XGBoost has many optimizations for calculating fast predictions, but when it comes to Intel CPUs, it is possible to substantially improve prediction times through other libraries. Here, we will demonstrate how to accelerate model serving using the [Model Builders](https://uxlfoundation.github.io/scikit-learn-intelex/latest/model_builders.html) module in the Extension for scikit-learn, which converts the fitted XGBoost model into a different format which is consumed by the [oneDAL](https://uxlfoundation.github.io/oneDAL/) library behind the scenes instead.

To re-calculate the new default probabilities with the optimized version, pass argument `--intel` to `run_inference.py`:

```shell
python src/run_inference.py --intel
```

* Example output without optimizations (XGBoost):

  ```
  INFO:root:First 5 predictions
  INFO:root:[[0.2856955  0.7143045 ]
   [0.721311   0.27868906]
   [0.3208146  0.6791854 ]
   [0.2856955  0.7143045 ]
   [0.2856955  0.7143045 ]]
  INFO:root:Inference time (XGBoost): 0.041802
  ```

* Example output with optimizations (Extension for scikit-learn):

  ```
  INFO:root:First 5 predictions
  INFO:root:[[0.28569542 0.71430458]
   [0.72131102 0.27868898]
   [0.32081456 0.67918544]
   [0.28569542 0.71430458]
   [0.28569542 0.71430458]]
  INFO:root:Inference time (Intel-optimized): 0.028231
  ```

### Optimized Reference Solution Implementation

Intel® optimizations for XGBoost, while initially only available through Intel-distributed builds of this library, have been upstreamed into the main release since XGBoost v0.81. As a result, by using a recent XGBoost version, you directly benefit from optimizations when running the code on a valid Intel® Architecture, without further need for any special configurations or special environments.

For inference, a trained XGBoost model can be further converted and served using the [oneDAL](https://uxlfoundation.github.io/oneDAL/) accelerator through the [Extension for scikit-learn](https://uxlfoundation.github.io/scikit-learn-intelex).

### Summary

Credit default prediction is a pivotal task to analyzing the financial risk that a particular obligation could bring to an organization. In this reference kit, we demonstrated a simple method to build an XGBoost classifier capable of predicting the probability that an issued loan will end in default (non-payment), which can be used continually as a component in real scenarios. Further, we also added some methods to introduce the concept of fairness and bias measurements and accounting for highly sensitive models such as this Credit Default Prediction for Lending. 

We also showed how to accelerate inference (model serving) for these models using Intel® optimizations available in the [Extension for scikit-learn](https://uxlfoundation.github.io/scikit-learn-intelex/latest/model_builders.html), which reduces infrastructure costs and allows a higher throughout of risk assessments.

## Notices & Disclaimers
Performance varies by use, configuration and other factors. Learn more on the [Performance Index site](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview/).<br>
Performance results are based on testing as of dates shown in configurations and may not reflect all publicly available updates. See backup for configuration details. No product or component can be absolutely secure. <br>
Your costs and results may vary. <br>
Intel technologies may require enabled hardware, software or service activation.<br>
© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others. <br>

To the extent that any public or non-Intel datasets or models are referenced by or accessed using tools or code on this site those datasets or models are provided by the third party indicated as the content source. Intel does not create the content and does not warrant its accuracy or quality. By accessing the public content, or using materials trained on or with such content, you agree to the terms associated with that content and that your use complies with the applicable license.
 
Intel expressly disclaims the accuracy, adequacy, or completeness of any such public content, and is not liable for any errors, omissions, or defects in the content, or for any reliance on the content. Intel is not liable for any liability or damages relating to your use of public content.
