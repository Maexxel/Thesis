from __future__ import annotations
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from starkweather import Trial

from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt


@dataclass
class MyTrial:
    """
    A class representing a single trial with relevant attributes.

    Attributes:
        rpes (List[float]): List of reward prediction errors (RPEs) for the trial.
        lenght (int): Total length of the trial.
        iti_lenght (int): Inter-trial interval length.
        isi_lenght (int): Inter-stimulus interval length.
        got_reward (bool): Whether the trial resulted in a reward.
        idx_in_trial (int): Index of the trial within the episode.
    """
    rpes: List[float]
    lenght: int
    iti_lenght: int
    isi_lenght: int
    got_reward: bool
    idx_in_trial: int

    def __post_init__(self):
        """
        Post-initialization checks to ensure data consistency.
        
        Raises:
            AssertionError: If the length of RPEs is not one less than the trial length.
            AssertionError: If the sum of ITI length and ISI length is not equal to the trial length minus one.
        """
        assert len(self.rpes) == self.lenght - 1, \
            f"Len of RPEs {len(self.rpes)} is not one smaller then trial-lenght {self.lenght} in trial {self.idx_in_trial}"
        assert self.iti_lenght + self.isi_lenght == self.lenght - 1, \
            f"ITI length and ISI length <{self.iti_lenght + self.isi_lenght}> doesn't add up to trial-length - 1 <{self.lenght}> in trial {self.idx_in_trial}"
        
def convert_to_mytrial(trials: List[Trial]) -> List[MyTrial]:
    """
    Convert a list of Trial objects to a list of MyTrial objects.
    
    Args:
        trials (List[Trial]): List of Trial objects to convert.
    
    Returns:
        List[MyTrial]: List of converted MyTrial objects.
    """
    def convert_single(trial: Trial) -> MyTrial:
        return MyTrial(rpes=trial.rpe,
                       lenght=trial.trial_length,
                       iti_lenght=trial.iti,
                       isi_lenght=trial.isi,
                       got_reward=trial.reward_size != 0,
                       idx_in_trial=trial.index_in_episode)
    
    return [convert_single(trial) for trial in trials]

@dataclass
class RpeGroup:
    """
    A class representing a group of RPEs (reward prediction errors) with associated calculations.

    Attributes:
        isi_lenght (None | int): Length of the inter-stimulus interval.
        rpes (List[npt.NDArray[np.float64]]): List of RPE arrays.
        rpes_avg (npt.NDArray[np.float64]): Array of average RPE values.
        group_max (None | float): Maximum value of the average RPEs.
    """
    isi_lenght: None | int
    rpes: List[npt.NDArray[np.float64]] = field(default_factory=lambda: [])
    rpes_avg: npt.NDArray[np.float64] = field(default_factory=lambda: [])
    group_max: None | float = None

    def calc_rpe_avg(self) -> None:
        """
        Calculate the average RPEs for the group.

        Raises:
            AssertionError: If the shape of the averaged RPEs does not match the ISI length.
        """
        self.rpes_avg = np.mean(np.vstack(self.rpes), axis=0)
        assert self.rpes_avg.shape == (self.isi_lenght,), \
            f"RPE averaging didn't work out. Original shape ({len(self.rpes)},{len(self.rpes[0])}) -> {self.rpes_avg.shape}, should be ({self.isi_lenght})"
    
    def calc_rpe_max(self) -> None:
        """
        Calculate the maximum value of the average RPEs.

        Raises:
            AssertionError: If RPEs are not provided or rpes_avg is not calculated.
        """
        assert self.rpes
        assert type(self.rpes_avg) is np.ndarray and self.rpes_avg.shape == self.isi_lenght, \
            "rpes_avg must be calculated before rpe_max. Use method <calc_rpe_avg>."
        self.group_max = np.max(self.rpes_avg)