import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# STREAMLIT

# The sidebar
st.sidebar.title("Books rating project")

pages = [
    "Introduction",
    "Data Exploration",
    "Feature Engineering",
    "Data Modeling",
    "Conclusion"]
page = st.sidebar.radio("", pages)

Team_members = [
    "Vixra KEO",
    "Margueritte Babeth NKEN",
    "Antoine BEDOUCH",
    "Attoumassa SAMAKE"]

# Team members
st.sidebar.markdown("---")
st.sidebar.title("Team members")
for member in Team_members:
    st.sidebar.markdown(member)

raw_df = pd.read_csv(
    filepath_or_buffer="report/data/books.csv",
    on_bad_lines="warn",
    sep=",",
    skipinitialspace=True)


# Page 0 : The project descrition
if page == pages[0]:
    st.title("Introduction")
    # st.image('gd2.jpeg')
    st.write("	Predictive analytics is a very broad set of practices aimed at analysing the data available to a company and making predictions on that data. It uses predictive models, supported by algorithms that learn from the data. Its areas of application are as diverse as the data available: in this case, we will focus on literature.")
    st.write("In this study, we aim to define and train a predictive model for book ratings. In particular, because book ratings are continuous values, the problem we are solving is called a regression model. For this purpose, we will use different models, compare them and choose the most optimal one.")
    st.write("This work will be divided into three parts:")
    st.write("- *Data Exploration*")
    st.write("- *Features Engineering*")
    st.write("- *Data Modeling*")



# Page 1: Analysis & cleaning
elif page == pages[1]:
    st.header("Data exploration")
    st.write("For this project we worked with data from the Goodreads website.")
    st.write("During the exploratory analysis, we first conducted a thorough investigation of our database. This involved observing the format of the dataset, seeing if there were any missing or incorrectly filled in data and choosing the appropriate processing method.")
    st.write("Then we proceded to the preliminary analysis (i.e. to assess the balance of the data) to determine the useful variables according to the information contained in these variables, and to eliminate those that are not very useful for our analysis.")

    
    # st.write(raw_df.head())
    if st.checkbox("**Preview Dataset **"):
        st.dataframe(raw_df)

    st.write(" The dataset has  11123 rows (thus as many book entries) and 12 columns including the target variable (average_rating).")
    st.markdown("      ")

    st.write("**Quantitative variables :**")
    st.write("- *Average_rating* : The average rating of the book received in total")
    st.write("- *num_pages*: The number of pages the book contains")
    st.write("- *ratings_count*: The total number of ratings the book received")
    st.write("- *text_reviews_count*: The total number of written text reviews the book received")

    st.write("**Qualitative variables:**")  
    st.write("- *title*: The name under which the book was published.")
    st.write("- *authors*: The names of the authors of the book. Multiple authors are delimited by “/”.")  
    st.write("- *language_code*: Indicates the primary language of the book. For instance, “eng” is standard for English.")
    st.write("- *publication_date*: The date the book was published.")
    st.write("- *publisher*: The name of the book publisher.")

    st.write("We also have some other variables for books identification: (*bookID*, *isbn*, *isbn13*).")



# NETTOYAGE DU JEU DE DONNÉES
    st.subheader("Clean up the dataset")
    # st.write("**Gestion des NaNs :**")
    st.write("Please note that there are no NA values.")
    # st.write("- Is the data balanced?")
    # st.write(" - Is there any outliers in our dataset ?")
    # st.write("- Which features are useful and which aren't?")

    st.write("We first plotted the distributions of the main numerical features:")
    st.write("- *number of pages*")
    st.write("- *number of ratings*")
    st.write("- *number of reviews*")
    st.write("- *average rating*")

    if st.checkbox("**Show plots**"):

        st.markdown("      ")
        fig_num_pages = plt.figure(figsize=[12, 5])
        df = raw_df
        plt.hist(df["num_pages"], bins=30)
        plt.yscale("log")
        plt.show()
        st.pyplot(fig_num_pages)
        st.caption("The distribution of number of pages ")

        st.markdown("      ")
        fig_num_rating = plt.figure(figsize=[12, 5])
        plt.hist(df["ratings_count"], bins=50)
        plt.yscale("log")
        plt.show()
        st.pyplot(fig_num_rating)
        st.caption("The distribution of number of ratings")

        st.markdown("      ")
        fig_text = plt.figure(figsize=[12, 5])
        plt.hist(df["text_reviews_count"], bins=50)
        plt.yscale("log")
        plt.show()
        st.pyplot(fig_text)
        st.caption("The distribution of number of reviews")

        st.markdown("      ")
        fig_ave = plt.figure(figsize=[12, 5])
        plt.hist(df["average_rating"], bins=50)
        plt.show()
        st.pyplot(fig_ave)
        st.caption("The distribution of average rating")

    st.write("We notice immediately that the target feature `average_rating` is skewed towards an average of 4. It resembles a normal distribution centered around 4. This data is inherently imbalanced (almost all ratings between 3 and 5, almost no ratings between 0 and 3).")
    st.write("We have observed that most books have less than 1000 pages. Under 1000 pages the distribution is overall balanced.")
    st.write("The distribution of number of ratings and reviews are skewed towards 0. This means that generally speaking, most books have fewer numbers of reviews and ratings while a few books have a lot. Such distribution resembles a [power law distribution](https://en.wikipedia.org/wiki/Power_law).")
    st.write("We will have to take this into consideration when creating the training and testing sets: these sets should both include books with a wide range of average ratings.")

    # st.write("Finally, we can plot the correlation matrix showing correlations between numerical values. This gives us an idea of how statistically correlated these variables are.")

    # df = raw_df
    # corr_matrix = df.corr()
    # fig_mat = plt.figure(figsize=(8, 7))
    # sns.heatmap(corr_matrix, annot=True)
    # plt.show()
    # st.pyplot(fig_mat)


elif page == pages[2]:
    df = raw_df
    df = df.drop(["bookID", "isbn", "isbn13"], axis=1)
    st.header("Feature Engineering")
    st.write("In this section we focused on leveraging techniques, in order to create new variables that were not verbatim in the original data. This helped us simplify and accelerate data transformations, while improving the model precision. The feature engineering was achieved in three steps.")

    st.write("We first proceeded to study the relationship existing between the selected variables of our dataset and our targeted variable. This was done with the use of a correlation matrix. Of all the chosen variables, we could see that only the number of pages had a better correlation to our targeted variable, average rating.")
    # begin code

    from datetime import datetime, MINYEAR, timedelta
    from time import strptime
    def date_to_datetime(date: str) -> datetime:
        """
        Converts date string to corresponding datetime format
        """
        try:
            new_time_format = datetime.strptime(date, "%m/%d/%Y")
            return new_time_format
        except ValueError:
            #  Some dates seem to be erronous (days that do not exist for given months)
            #  Correcting these days manually
            newdate = date.split("/")
            if newdate[1] == "31":
                newdate[1] = "30"
            newdate = newdate[0] + "/" + newdate[1] + "/" + newdate[2]
            return datetime.strptime(newdate, "%m/%d/%Y")

    df["publication_date"] = df["publication_date"].apply(date_to_datetime)
    oldest_book = min(df["publication_date"])
    newest_book = max(df["publication_date"])

    print(f"Oldest book published on {oldest_book}.")
    print(f"Most recent book published on {newest_book}")

    def normalise_age(book_date: datetime) -> float:
        """
        Converts book date into normalised value in [0, 1] interval where 0 corresponds to the oldest book and 1 corresponds to most recent book
        """
        return (book_date.timestamp() - oldest_book.timestamp()) / (newest_book.timestamp() - oldest_book.timestamp())


    def get_real_age(normalised_age: float) -> datetime:
        """
        Get back the date from a normalised age.
        """
        datetime_s = oldest_book.timestamp() + (newest_book.timestamp() - oldest_book.timestamp()) * normalised_age
        return datetime.fromtimestamp(datetime_s)

    df["normalised_age"] = df["publication_date"].apply(normalise_age)

    corr_matrix = df.corr()
    fig_mat = plt.figure(figsize=(8, 7))
    sns.heatmap(corr_matrix, annot=True)
    plt.show()
    st.pyplot(fig_mat)
    # end code
    st.write("In this way, we could get a first overview of which variables explain the average rating obtained by a book, and more likely help us predict potential average book ratings; nb_ratings, nb_reviews and nb_pages. ")

    
    st.write("**1- Defining training variable features**")
    st.write("This was achieved through the creation of a function, the *normalise_age* function, that converts a date to number of seconds since a reference time. The function then normalises this number of seconds between 0 and 1 values, where 0 corresponds to the oldest book in our dataset and 1 the most recently published book. This allowed us carry out an age distribution evaluation. We noticed was that the distribution was imbalanced, skewing towards more recent books.")


    st.write("**2- Evaluating Other Variables**")
    # st.write("Having an idea of the main variables of our model, it is, however, still necessary to evaluate other variables that could bring more information to our model. Thus we proceeded to analysing the different languages in which were written the books, the book titles, publishers and authors.")
    st.write("Then we proceeded to analysing the different languages in which were written the books, the book titles, publishers and authors.")
    st.write("By so doing, it could be observed that most books were written in English, and only *5.24%* of all the books were written in foreign languages. We then created a binary feature for whether the book was written in english (0), or in a foreign language (1)")
    st.write("There are a total of 6639 unique authors, 2290 unique publishers and 10348 unique titles. This information would need to be transformed into numerical values if we wanted to use it as features. Thousands of names cannot be converted into quantifiable measures in a realistic manner. Thus, we did not find it interesting to use these variables.")


    st.write("**3- Handling Outliers**")
    st.write("Outliers are values that are unusual in your dataset and might affect statistical analysis by challenging their presumptions. To avoid this, it is important to handle them prior to any analysis.")
    st.write("The first step was detecting them using a boxplot. Then, we went on to filter the outliers by setting up thresholds (books with more than 2e3 number of pages for instance), after which we observe the number outliers we have above that scale and delete them. We repeated the same procedure for the ratings count, the text reviews count and the normalised age. Of course, by setting up rules based on what was initially observed from the boxplot.")

    st.write("At the end we kept 96.27% of data.")


elif page == pages[3]:
    df = raw_df
    st.header("Data Modeling")
    st.write("Based on the feature engineering, we decided to keep the following predictor variables: ")
    st.write("- *num_pages*")
    st.write("- *ratings_count*")
    st.write("- *text_reviews_count*")
    st.write("- *normalized_age*")
    st.write("- *language*")

    st.write("The variable to be predicted is *average_rating*.")

    st.write("**1. Data Splitting**")
    st.write("Prior to the modeling, we split our data into two subsets; a 20% subset for test and the remaining 80% for training. We then went on to compare the two subsets through a histogram, and we could observe a resemblance between both subsets and the original distribution. ")
    st.write("This means that even though the target feature in imbalanced, both the training and testing sets have similar distributions")
    # {Insert the training data and test data histogram}

    # begin code
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, max_error, mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    df_train, df_test =  train_test_split(df, test_size = 0.2, random_state=42)
    fig_test = plt.figure(figsize=(8, 7))
    plt.hist(df_train["average_rating"], bins=50, label="training data", alpha=0.5)
    plt.hist(df_test["average_rating"], bins=50, label="testing data", alpha=0.5)
    plt.legend()
    plt.show()
    st.pyplot(fig_test)

    # end code

    st.write("2. **Model Evaluation**")
    st.write("It is helpful to consider which method to use in the evaluation of our models before we train the model. The ones we used are the Root-Mean-Square Error (RMSE) and the coefficient of determination (R-square or R²), both being the most popular metrics for regression models.")

    st.write("Looking at the maximum error, or the prediction that performed the poorest, can also be helpful. With this is mind, we developed a function to evaluate the model to help compare several models. It accepts the actual and expected scores and performs the following actions:")
    st.write("- Compares the prediction of the 20 first books in our test dataframe with the actual values in a bar plot.")
    st.write("- Plots a scatterplot of the predicted ratings as a function of the true ratings.")
    st.write("- Computes, prints, and returns the following metrics: Max error, RMSE and R² score")

    st.write("3. **Modeling**")
    st.write("We carried out 7 different models	, in order to compare them and evaluate which of the 7 is the most optimal, the 7 models being:")
    st.write("- Linear regression")
    st.write("- Random Forest")
    st.write("- Decision Tree")
    st.write("- Support Vector Regression")
    st.write("- Gradient Boosting")
    st.write("- Adaboost")
    st.write("- StackingCV")

    st.write("Here are the summarised results of each model and for each metric:")

    all_evaluations = pd.read_csv("report/data/evaluations.csv")

    all_evaluations

    st.write("After carrying out the modeling made the following observations: ")
    st.write("First, we note the correlation between the RMSE and the R² score. In general, the better the R² (lower), the better (higher) the RMSE. This suggests that at the very least, these measurements are consistent with one another. Also, we observe that while the R² score displays significantly more diverse values, the RMSE scores are often comparable to one another (0.3 < RMSE < 0.4).")
    st.write("Secondly, we notice that the Gradient Boosting is the model that performs the best in terms of R² score and RMSE. However, the Adaboost Decision Tree model performs the best if minimizing the maximum error is preferred. In other words, compared to models like Gradient Boosting and Random Forest that often perform better, its worst errors are less incorrect.")


elif page == pages[4]:
    st.header("Conclusion")
    st.write("This project consisted of defining a model for predicting the ratings of a set of books. For this we used the Bookreads dataset.")

    st.write("We first carried out exploratory analysis, with the aim of getting to know our database, its flaws and the set of variables that were in it. We proceeded to the treatment of missing data, descriptive statistics, visualization of correlation coefficients and the choice of variables for the model.")
    st.write("After that we went on with the feature engineering, in which we did an evaluation of the remaining variables and handled outliers. Finally, we carried out the modeling. At the end of the modeling, it could be deduced that the Gradient Boost model performed best with respect to the R² and RMSE.")

    st.write("**Limits:**")
    st.write("- Removing outliers simplified the training but seemed to make the prediction more dependant on input data (hint at overfitting)")
    st.write("- r and least squares regression are NOT resistant to outliers.")
    st.write("- Other factors than the predictor variables that are not studied but have an influence on the response variable may exist. In other words, the data we have doesn't fully explain the results. This was to be expected due to the low values in the correlation matrix.")
    st.write("- A high degree of correlation does not imply causality and effect. Maybe the features our models trained do not have real predictive power and the model is just learning from correlation. If this was the case, then our model wouldn't be generalisable to make predictions on other data.")

    st.write("**Suggestions:**")
    st.write("- Employing SMOTE data augmentation due to unbalanced dataset in terms of average_rating.")
    st.write("- Enriching the data by merging fresh data (using ISBN as merging key).")
    st.write("- Using NLP to exploit the dropped characteristics such as book titles.")
