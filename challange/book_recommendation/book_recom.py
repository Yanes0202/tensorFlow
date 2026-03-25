import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

books_path = '../../resources/book/BX-Books.csv'
ratings_path = '../../resources/book/BX-Book-Ratings.csv'

df_books = pd.read_csv(
    books_path,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_path,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

user_counts = df_ratings['user'].value_counts()

book_counts = df_ratings['isbn'].value_counts()

df_ratings_filtered = df_ratings[
    df_ratings['user'].isin(user_counts[user_counts >= 200].index) &
    df_ratings['isbn'].isin(book_counts[book_counts >= 100].index)
    ]

df_combined = pd.merge(df_ratings_filtered, df_books, on='isbn')
df_combined = df_combined.drop_duplicates(['user', 'title'])

book_pivot = df_combined.pivot(index='title', columns='user', values='rating').fillna(0)
book_sparse = csr_matrix(book_pivot.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(book_sparse)


def get_recommends(book=""):
    if book not in book_pivot.index:
        return [book, ["Book is not existing in the database", 0]]

    book_vector = book_pivot.loc[book].values.reshape(1, -1)

    distances, indices = model_knn.kneighbors(book_vector, n_neighbors=6)

    recommend_list = []
    for i in range(1, 6):
        index = indices.flatten()[i]
        dist = float(distances.flatten()[i])
        recommend_list.append([book_pivot.index[index], dist])

    recommended_books = [book, recommend_list[::-1]]

    return recommended_books


books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)


def test_book_recommendation():
    test_pass = True
    recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
    if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
        test_pass = False
    recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
    recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
    for i in range(4):
        if recommends[1][i][0] not in recommended_books:
            test_pass = False
        if abs(float(recommends[1][i][1]) - recommended_books_dist[i]) >= 0.05:
            test_pass = False
    if test_pass:
        print("You passed the challenge! 🎉🎉🎉🎉🎉")
    else:
        print("You haven't passed yet. Keep trying!")


test_book_recommendation()
