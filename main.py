from surprise import Reader, Dataset, SVDpp, accuracy
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd

users_data = "BX-Users.csv"
books_data = "BX-Books.csv"
ratings_data = "BX-Books-Ratings.csv"


users = pd.read_csv("BX-Users.csv", sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']

books_column = ['']

rating = pd.read_csv("BX-Book-Ratings.csv", sep=';', error_bad_lines=False, encoding='latin-1', low_memory=False)
rating.columns = ['userID', 'ISBN', 'bookRating']

df = pd.merge(users, rating, on='userID', how='inner')
df.drop(['Location', 'Age'],
        axis=1,
        inplace=True)

#print(df.head())
min_book_ratings = 50
filter_books = df['ISBN'].value_counts() > min_book_ratings
filter_books = filter_books[filter_books].index.tolist()

min_user_ratings = 50
filter_user = df['userID'].value_counts() > min_user_ratings
filter_user = filter_user[filter_user].index.tolist()

df_new = df[(df['ISBN'].isin(filter_books)) & (df['userID'].isin(filter_user))]

reader = Reader(rating_scale=(0, 9))
data = Dataset.load_from_df(df_new[['userID', 'ISBN', 'bookRating']], reader)
algo = SVDpp()

trainSet, testSet = train_test_split(data, test_size=.20)
#algo.fit(trainSet)
#pred = algo.test(testSet)
#Accuracy = accuracy.rmse(pred)

#print("Accuracy of SVDpp is", Accuracy)

print("Dataset Size", df.size)
print("Usable Dataset Size", df_new.size)
#print(type(trainSet), trainSet.n_items())
