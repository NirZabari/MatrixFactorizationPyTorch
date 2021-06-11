import argparse
from tqdm import tqdm
from utils import *

def evaluate_test(item_embeddings, user_embeddings, test_df):
    def calc_rating(rating):
        return (item_embeddings[rating["movieId"]].detach() * user_embeddings[rating["userId"]].detach()).sum().item()

    test_df = test_df.assign(predicted_score=test_df.apply(calc_rating, axis=1))
    test_df = test_df.assign(mse=(test_df["predicted_score"] - test_df["rating"]) ** 2)

    return test_df["mse"].sum()

def train(latent_dimension=40, num_epochs=20, lr=0.1, batch_size=64, criterion=torch.nn.MSELoss()):
    dataloader = get_dataloader(batch_size=batch_size)
    users_embedding, items_embedding = init_embedding_vectors(dataloader, latent_dimension)

    loss_epochs = {}
    for epoch in range(num_epochs):
        train_loss = 0
        tqdm_iter = tqdm(enumerate(iter(dataloader)))

        for i, batch in tqdm_iter:
            items_ids = batch[dataloader.dataset.MOVIE_ID].to(int).tolist()
            users_ids = batch[dataloader.dataset.USER_ID].to(int).tolist()

            ratings = batch[dataloader.dataset.RATING].float()

            batch_user = get_embedding_from_dict(d=users_embedding, keys=users_ids)
            batch_items = get_embedding_from_dict(d=items_embedding, keys=items_ids)
            [t.requires_grad_(True) for t in batch_user + batch_items]
            optimizer = torch.optim.SGD(batch_items + batch_user, lr=0.1)
            optimizer.zero_grad()

            batch_user_embbedding = torch.stack(batch_user)[:, 0, :]
            batch_item_embbedding = torch.stack(batch_items)[:, 0, :]

            res_multiplication = batch_user_embbedding @ batch_item_embbedding.T
            predicted_score = torch.einsum('ii -> i', res_multiplication)

            loss = criterion(predicted_score, ratings)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * dataloader.batch_size
            tqdm_iter.set_description_str(f"loss: {loss.item()} ")
            update_embeddings_to_dict(d=items_embedding, keys=items_ids, new_values=batch_items)
            update_embeddings_to_dict(d=users_embedding, keys=users_ids, new_values=batch_user)


        test_df = dataloader.dataset.get_test_ratings()
        test_loss = evaluate_test(items_embedding, users_embedding, test_df)
        loss_epochs[epoch] = {'train_loss': train_loss / dataloader.dataset.get_train_ratings_size(),
                              'test_loss': test_loss / dataloader.dataset.get_test_ratings_size()}

    plot_loss(loss_epochs, f"loss (latent dim: {latent_dimension} lr: {lr} batch_size: {batch_size})")

if __name__ == "__main__":
    pargs = argparse.ArgumentParser(description='MF')
    pargs.add_argument('--lr', default=0.01, type=float)
    pargs.add_argument('--latent_dimension', default=10, type=int)
    pargs.add_argument('--batch_size', default=256, type=int)
    pargs.add_argument('--num_epochs', default=30, type=int)
    train(**pargs.parse_args().__dict__)
