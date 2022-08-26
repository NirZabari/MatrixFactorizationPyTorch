import argparse
from easydict import EasyDict as edict
from tqdm import tqdm
from utils import *
import ray
from ray import tune
from functools import partial

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)


def evaluate_test(item_embeddings, user_embeddings, test_df):
    def calc_rating(rating):
        return (item_embeddings[rating["movieId"]].detach() * user_embeddings[rating["userId"]].detach()).sum().item()

    test_df = test_df.assign(predicted_score=test_df.apply(calc_rating, axis=1))
    test_df = test_df.assign(mse=(test_df["predicted_score"] - test_df["rating"]) ** 2)
    test_df = test_df.assign(is_rating_correct=(test_df.predicted_score - test_df.rating).abs() < 0.5)
    mse = test_df["mse"].sum()
    accuracy = test_df.is_rating_correct.sum() / len(test_df)
    tune.report({'mse': mse, 'accuracy': accuracy})
    return mse, accuracy


def train(latent_dimension=40, num_epochs=20, lr=0.1, batch_size=64, criterion=torch.nn.MSELoss(), **kwargs):
    dataloader = get_dataloader(batch_size=batch_size)
    size_dataloader = len(dataloader)
    users_embedding, items_embedding = init_embedding_vectors(dataloader, latent_dimension)

    loss_epochs = {}
    for epoch in range(num_epochs):
        train_loss = 0
        tqdm_iter = tqdm(enumerate(iter(dataloader)))

        for i, batch in tqdm_iter:
            items_ids = batch[dataloader.dataset.MOVIE_ID].to(int).tolist()
            users_ids = batch[dataloader.dataset.USER_ID].to(int).tolist()

            ratings = batch[dataloader.dataset.RATING].float().to(device)

            batch_user = get_embedding_from_dict(d=users_embedding, keys=users_ids)
            batch_items = get_embedding_from_dict(d=items_embedding, keys=items_ids)
            [t.requires_grad_(True) for t in batch_user + batch_items]
            optimizer = torch.optim.SGD(batch_items + batch_user, lr=0.1)
            optimizer.zero_grad()

            batch_user_embbedding = torch.stack(batch_user)[:, 0, :].to(device)
            batch_item_embbedding = torch.stack(batch_items)[:, 0, :].to(device)

            res_multiplication = batch_user_embbedding @ batch_item_embbedding.T
            predicted_score = torch.einsum('ii -> i', res_multiplication)

            loss = criterion(predicted_score, ratings)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * dataloader.batch_size
            if i % 5000 == 0 or True:
                tqdm_iter.set_description_str(
                    f"loss: {loss.item()} ({i} / {size_dataloader} - {100 * i / size_dataloader:.3f}%)")
            update_embeddings_to_dict(d=items_embedding, keys=items_ids, new_values=batch_items)
            update_embeddings_to_dict(d=users_embedding, keys=users_ids, new_values=batch_user)

        test_df = dataloader.dataset.get_test_ratings()
        test_loss, accuracy = evaluate_test(items_embedding, users_embedding, test_df)
        loss_epochs[epoch] = {'train_loss': train_loss / dataloader.dataset.get_train_ratings_size(),
                              'test_loss': test_loss / dataloader.dataset.get_test_ratings_size(),
                              'accuracy': accuracy}

    print(loss_epochs)
    print("finished run")
    # plot_loss(loss_epochs, f"loss (latent dim: {latent_dimension} lr: {lr} batch_size: {batch_size})")


def get_args():
    pargs = argparse.ArgumentParser(description='MF')
    pargs.add_argument('--lr', default=0.01, type=float)
    pargs.add_argument('--latent_dimension', default=10, type=int)
    pargs.add_argument('--batch_size', default=256, type=int)
    pargs.add_argument('--num_epochs', default=10, type=int)
    pargs.add_argument('--hpo', action='store_true')
    arguments = pargs.parse_args()
    return edict(arguments.__dict__)


def hpo_search(config, args):
    args.update(config)
    train(**args)


if __name__ == "__main__":
    args = get_args()

    if args.hpo:
        from hpo import run_hpo, RayTuneModes

        # we freeze all hyper-parameters as input, and later ray will update its paramater  with the config
        hpo_search_func = partial(hpo_search, args=vars(args))
        for mode in [RayTuneModes.ASHS, RayTuneModes.MEDIAN_STOPPING_RULE, RayTuneModes.AX, RayTuneModes.OPTUNA]:
            num_samples = 20
            print(f"=================\ncurrent mode = {mode}\n=================")
            run_hpo(train_func=hpo_search_func,
                    mode=mode,
                    num_samples=num_samples)
        print("finish HPO process succefully")
    else:  # normal run
        train(**args)
