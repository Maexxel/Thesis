from typing import List
import numpy as np
from torch.utils.data import Dataset


class AlternatingDataset(Dataset):
    """
    A custom dataset where each sequence alternates between 0 and 1.
    Each sequence starts randomly with a 0 or 1.
    Goal is to predict the next value in the sequence.
    """
    def __init__(self) -> None:
        xs_helper = np.random.randint(0,2,10)
        self.xs = [[[k%2] for k in range(xs_helper[i], 5 + xs_helper[i])] for i in range(10)]
        self.ys = [[[1 - x[-1]] for x in x_com] for x_com in self.xs]

    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


class MoreThen3Dataset(Dataset):
    """
    A custom dataset where each sequence has labels based on the
    sum of elements being greater than or equal to 3.
    """
    def __init__(self) -> None:
        self.xs = np.random.randint(0, 2, (100, 10, 1))
        self.ys = [[[1 if np.sum(seq[:idx]) >= 3 else 0] for idx in range(len(seq))] for seq in self.xs]

    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

from starkweather import Starkweather
class MyStarkweather(Dataset):
    @staticmethod
    def init_from_disk(path: str) -> 'MyStarkweather':
        s = MyStarkweather(None, None, None, None, None, True)
        with open(f"{path}_xs.npy", "rb") as f:
            s.xs = np.load(f)
        with open(f"{path}_ys.npy", "rb") as f:
            s.ys = np.load(f)

        return s

    def __init__(self,
                 n_sequences: int,
                 len_sequences: int,
                 omission_probability: float,
                 iti_p: float,
                 iti_min: int,
                 _clear_init: bool = False) -> None:
        if _clear_init:
            return

        trials_per_exp = len_sequences // 14 # is a lower limit

        self.raw_exps: List[Starkweather] = []
        self.xs: np.ndarray = np.empty((n_sequences, len_sequences, 2))
        self.ys: np.ndarray = np.empty((n_sequences, len_sequences, 1))

        total_xs_count: int = 0
        for seq_idx in range(n_sequences):
            raw_exp = Starkweather(ncues=1,
                                   ntrials_per_cue=trials_per_exp,
                                   ntrials_per_episode=trials_per_exp,
                                   omission_probability=omission_probability,
                                   iti_p=iti_p, iti_min=iti_min, t_padding=0)
            total_xs_count += sum([trial.trial_length for trial in raw_exp.trials])
            self.raw_exps.append(raw_exp)
            self.xs[seq_idx] = raw_exp[0][0].numpy()[:len_sequences]
            self.ys[seq_idx] = raw_exp[0][1].numpy()[:len_sequences]

        self.XS_USAGE = n_sequences * len_sequences / total_xs_count

    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


from rl_env.environment import runAgent
class RLDataset(Dataset):
    def __init__(self, agent, env, n_sequences, len_sequences, _clear_init=False) -> None:
        if _clear_init:
            return

        self.xs = np.zeros((n_sequences, len_sequences, 2))
        self.ys = np.zeros((n_sequences, len_sequences, 1))

        for s_idx in range(n_sequences):
            choices, rewards, _, _ = runAgent(env, agent, len_sequences)

            prev_choices = np.concatenate(([0], choices[0:-1]))
            prev_rewards = np.concatenate(([0], rewards[0:-1]))

            self.xs[s_idx] = np.swapaxes(
                np.concatenate(([prev_choices], [prev_rewards]), axis=0), 0, 1)
            self.ys[s_idx] = np.expand_dims(choices, 1) # same as reshape(-1,1)

    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


class SimpleDataset(Dataset):
    def __init__(self, xs, ys) -> None:
        self.xs = xs
        self.ys = ys
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

def to_disk(dataset: Dataset, path: str) -> None:
    xs_path = f"{path}_xs.npy"
    ys_path = f"{path}_ys.npy"

    with open(xs_path, "wb") as f:
        np.save(f, dataset.xs)
    with open(ys_path, "wb") as f:
        np.save(f, dataset.ys)

    print(f"Xs (shape={dataset.xs.shape} was saved to : <{xs_path}>.")
    print(f"Ys (shape={dataset.ys.shape} was saved to : <{ys_path}>.")


def from_disk(dataset: str, path: str) -> None:
    match dataset:
        case "MyStarkweather":
            s = MyStarkweather(None, None, None, None, None, True)
        case "RLDataset":
            s = RLDataset(None, None, None, None, True)
        case _:
            print(f"Dataset <{dataset}> is not supported.")
            return
        
    with open(f"{path}_xs.npy", "rb") as f:
        s.xs = np.load(f)
    with open(f"{path}_ys.npy", "rb") as f:
        s.ys = np.load(f)

    return s

def custom_collate(data):
    """
    Custom collate function to combine a list of samples into a batch.

    Args:
        data (list): A list of tuples where each
            tuple contains a sequence and its corresponding label.

    Returns:
        tuple: A tuple containing two numpy arrays,
        one for the sequences and one for the labels.
    """
    x_batch = []
    y_batch = []
    for x, y in data:
        x_batch.append(x)
        y_batch.append(y)
    return np.array(x_batch), np.array(y_batch)

if __name__ == "__main__":
    from rl_env.environment import RandomWalkEnv
    from rl_env.agents import QAgent
    from multiprocessing import Pool, cpu_count

    def create_dataset(args):
        agent, env, n_sequences, len_sequences = args
        return RLDataset(agent, env, n_sequences, len_sequences)

    def generate_datasets(agent, env, n_sequences, len_sequences, num_datasets):
        args = [(agent, env, n_sequences, len_sequences) for _ in range(num_datasets)]
        with Pool(processes=num_datasets) as pool:
            datasets = pool.map(create_dataset, args)
        return datasets

    np.random.seed(0)
    num_datasets = 100  # Number of datasets to create
    n_sequences = 1000
    len_sequences = 3000

    agent = QAgent()
    env = RandomWalkEnv()
    
    datasets = generate_datasets(agent, env, n_sequences, len_sequences, num_datasets)

    for i, dataset in enumerate(datasets):
        print(f"Dataset {i} length: {len(dataset)}")