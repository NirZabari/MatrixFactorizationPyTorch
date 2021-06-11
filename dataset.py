from torch.utils.data import Dataset
import pandas as pd
from os.path import join

class MovieLensDataSet(Dataset):
    USER_ID = "userId"
    MOVIE_ID = "movieId"
    RATING = "rating"

    def __init__(self, movie_lens_dir="./movielens-100k"):
        self.movie_lens_dir = movie_lens_dir
        self.load_df(path=self.movie_lens_dir)

    def load_df(self, path, train_ratio=0.8):
        self.links_df = pd.read_csv(join(path, 'links.csv'))
        self.movies_df = pd.read_csv(join(path, "movies_v2.csv"))
        self.ratings_df = pd.read_csv(join(path, "ratings.csv"))
        self.movies_df = self.movies_df[self.movies_df[self.MOVIE_ID].isin(self.ratings_df[self.MOVIE_ID])]

        #train-test split
        self.ratings_df = self.ratings_df.sample(frac=1)
        num_ratings = len(self.ratings_df)
        num_train = int(train_ratio * num_ratings)

        self.train_ratings_df = self.ratings_df.iloc[:num_train]
        self.test_ratings_df = self.ratings_df.iloc[num_train:]

        print(f"number of ratings in train-set: {len(self.train_ratings_df)}")
        print(f"number of ratings in test-set: {len(self.test_ratings_df)}")

        for c in [self.USER_ID, self.MOVIE_ID]:
            self.ratings_df[c] = self.ratings_df[c].astype(int)
        self.tags_df = pd.read_csv(join(path, "tags.csv"))

    def __len__(self):
        return len(self.train_ratings_df)

    def __getitem__(self, item):
        return self.train_ratings_df.iloc[item][[self.USER_ID, self.MOVIE_ID, self.RATING]].to_dict()

    def get_train_ratings_size(self):
        return len(self.train_ratings_df)

    def get_test_ratings_size(self):
        return len(self.test_ratings_df)

    def get_test_ratings(self):
        return self.test_ratings_df

    def get_unique_users(self):
        return self.ratings_df.userId.unique().tolist()

    def get_unique_items(self):
        return self.movies_df.movieId.unique().tolist()
