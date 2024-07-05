import numpy as np


class RandomWalkEnv:
    def __init__(self,
                 sigma: float = 0.15,
                 n_actions: int = 2,
                 upper_bound: int = 1,
                 lower_bound: int = 0
                 ) -> None:
        
        assert sigma > 0, "sigma should be greater then 0"
        assert lower_bound < upper_bound, "lower bound should be smaller then upper bound"
        
        self._sigma = sigma
        self._n_actions = n_actions
        self._upper_bound: int = upper_bound
        self._lower_bound: int = lower_bound
        
        self._new_session()

    def _new_session(self):
        # Pick new reward probabilities.
        # Sample randomly between 0 and 1
        self._reward_probs = np.random.rand(self._n_actions)
        
    def step(self, choice: int) -> int:
        assert choice < self._n_actions, f"The Environment has only {self._n_actions} actions, choice must be in that range"
        
        # Sample reward with the probability of the chosen side
        reward = np.random.rand() < self._reward_probs[choice]

        # Add gaussian noise to reward probabilities
        self._reward_probs = np.random.normal(loc=self._reward_probs,
                                              scale=self._sigma)

        self._reward_probs = np.clip(self._reward_probs, 0, 1)

        return reward
    
def runAgent(env,
             agent,
             n_trials: int = 200,
             n_qs: int = 2):
    choices = np.zeros(n_trials)
    rewards = np.zeros(n_trials)
    qs = np.zeros((n_trials, n_qs))
    reward_probs = np.zeros((n_trials, 2))
    
    # For each trial: Step the agent, step the environment, record everything
    for trial_i in np.arange(n_trials):
        # Record environment reward probs and agent Qs
        reward_probs[trial_i] = env._reward_probs
        qs[trial_i, :] = agent.q
        # First, agent makes a choice
        choice = agent.get_choice()
        # Then, environment computes a reward
        reward = env.step(choice)
        # Finally, agent learns
        agent.update(choice=choice, reward=reward)
        # Log choice and reward
        choices[trial_i] = choice
        rewards[trial_i] = reward
    
    return choices, rewards, qs, reward_probs