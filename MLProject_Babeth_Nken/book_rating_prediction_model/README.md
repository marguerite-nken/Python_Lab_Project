# Book Rating regression project 

## 1. Project Description

The objective of this Machine Learning project is to predict a book’s rating based on a dataset coming from the Goodreads website, a real readers community.

Predicting the rating of a book can be approached as a regression problem because it involves predicting a continuous numerical value: in this case the average rating. For that reason, we can use a Supervised learning method, allowing us to predict a continuous output variable (the book rating) based on one or more input variables (such as titles, authors, number of page, ratings count, publishers and many more).

After processing the data and engineering the features, we have trained a handful of different regression models. Finally, we have evaluated and compared these models.

## 2. How to Install and Run the Project


The project requires the following to be installed. 
- [python3.9](https://www.python.org/downloads/)
- [git](https://git-scm.com/)

Optionally, for easier virtual environment creation:
- [anaconda](https://www.anaconda.com/)

#### Instructions

Clone the repository:
```shell
git clone https://github.com/Attoumassa/book_rating_prediction_model.git
cd book_rating_prediction_model
git checkout main
```

Create and activate python virtual environment:

- Using conda (preferred method)
```shell
conda create -n "book_rating_prediction_model_env_aamv" python=3.9
conda activate book_rating_prediction_model_env_aamv
```

- Using venv
```shell
# Create the virtual environment
python3.9 -m venv book_rating_prediction_model_env_aamv

# Activate the virtual environment
## UNIX-like systems
source book_rating_prediction_model_env_aamv/bin/activate

## Windows with powershell
.\book_rating_prediction_model_env_aamv\Scripts\Activate.ps1
```

Installing package dependencies
```shell
pip install -r requirements.txt
```


## 3. How to Use the Project

The project consists of a jupyter notebook. Simply run each cell and read the comments. 

**How to open the notebook:**

```shell
jupyter notebook src/final_notebook.ipynb
```

**How to open the interactive report:**

```shell
python report/report.py
streamlit run report/report.py
```

Note: If the interactive version of the report doesn't work on your computer for whichever reason, a static export has been made and can be found in the following pdf file: `report/report.pdf`


The project is structured as follow:

- Introduction

- Data exploration
    - Loading the data
    - Preliminary analysis
    - Removing unused data

- Feature Engineering
    - Publication date
    - Language
    - Title, publisher and author data
    - Outliers
    - Conclusion
- Data modeling
    - Splitting the data
    - How to evaluate a model
    - Linear Regression
    - Random Forest
    - Decision Tree Regressor
    - Support Vector Regression (SVR) with Radial Basis Function (RBF) kernel
    - Gradient Boosting
    - Adaboost
    - Stacking

- Comparing the models
    - Modeling score comparison
    - Comparing model with all features vs less features
    - Comparing with and without outliers

- Conclusion

## 4. Credits

This project has been made by:
- [Attoumassa Samaké](https://github.com/Attoumassa)
- [Marguerite Nken](https://github.com/marguerite-nken)
- [Vixra Keo](https://github.com/Vixk2021)
- [Antoine Bedouch](https://github.com/Antoine-bdc)