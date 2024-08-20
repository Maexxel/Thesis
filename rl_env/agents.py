import numpy as np

class QAgent:
  """An agent that runs simple Q-learning for the y-maze tasks.

  Attributes:
    alpha: The agent's learning rate
    beta: The agent's softmax temperature
    q: The agent's current estimate of the reward probability on each arm
  """
  def __init__(self,
               alpha: float = 0.3,
               beta: float = 3.,
               n_actions: int = 2,):
    """Update the agent after one step of the task.

    Args:
      alpha: scalar learning rate
      beta: scalar softmax inverse temperature parameter.
      n_actions: number of actions (default=2)
      forgetting_rate: rate at which q values decay toward the initial values (default=0)
      perseveration_bias: rate at which q values move toward previous action (default=0)
    """
    self._prev_choice = None
    self._alpha = alpha
    self._beta = beta
    self._n_actions = n_actions
    self._q_init = 0.5
    self.new_sess()

    self._check_in_0_1_range(alpha, 'alpha')

  def new_sess(self):
    """Reset the agent for the beginning of a new session."""
    self._q = self._q_init * np.ones(self._n_actions)

  def get_choice(self) -> int:
    """Sample a choice, given the agent's current internal state."""
    decision_variable = self._beta * self._q
    choice_probs = np.exp(decision_variable) / np.sum(np.exp(decision_variable))
    return np.random.choice(self._n_actions, p=choice_probs)

  def update(self,
             choice: int,
             reward: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    self._q[choice] = (1 - self._alpha) * self._q[choice] + self._alpha * reward

  @property
  def q(self):
    # This establishes q as an externally visible attribute of the agent.
    # For agent = AgentQ(...), you can view the q values with agent.q; however,
    # you will not be able to modify them directly because you will be viewing
    # a copy.
    return self._q.copy()

  def _check_in_0_1_range(self,x, name):
    if not (0 <= x <= 1):
      raise ValueError(f'Value of {name} must be in [0, 1] range. Found value of {x}.')

class LeakyActorCriticAgent:
  def __init__(self, v:float = .5, theta:float = .0, alpha_v=0.3, alpha_l=2, alpha_f=0.05) -> None:
    self.v = v
    self.theta = theta

    self.alpha_v = alpha_v
    self.alpha_l = alpha_l
    self.alpha_f = alpha_f
    
  def update(self, reward: int, choice: int):
    """
    reward: 0 (no reward) or 1 (reward), used to update the two latent variables
    """
    def logistic(x: float) -> float:
          return 1 / (1 + np.exp(-x))
    
    if choice == 1:
        self.theta = (1 - self.alpha_f) * self.theta + self.alpha_l * (reward - self.v) * (1 - logistic(self.theta)) 
    else:
        self.theta = (1 - self.alpha_f) * self.theta - self.alpha_l * (reward - self.v) * logistic(self.theta)

    self.v = (1 - self.alpha_v) * self.v + self.alpha_v * reward

  def get_choice(self):
    """
    Calculates and returns choice, also updates internal choice variable.
    """
    choice_probs = 1 / (1 + np.exp(-self.theta))
    return np.random.choice(2, p=[choice_probs, 1-choice_probs])
  
  @property
  def q(self):
    return np.array([self.theta, self.v])
  
class LeakyActor2:
  """An agent that runs simple Q-learning for the y-maze tasks.

  Attributes:
    alpha: The agent's learning rate
    beta: The agent's softmax temperature
    q: The agent's current estimate of the reward probability on each arm
  """
  def __init__(self,
               alpha_l: float = 1.,
               alpha_v: float = 0.3,
               alpha_f: float = 0.05,
               n_actions: int = 2,):
    """Update the agent after one step of the task.

    Args:
      alpha: scalar learning rate
      beta: scalar softmax inverse temperature parameter.
      n_actions: number of actions (default=2)
      forgetting_rate: rate at which q values decay toward the initial values (default=0)
      perseveration_bias: rate at which q values move toward previous action (default=0)
    """
    self._alpha_v = alpha_v
    self._alpha_l = alpha_l
    self._alpha_f = alpha_f

    self.v = 0

    self._n_actions = n_actions
    assert self._n_actions == 2, "This Leaky Impl only works with n_actions == 2"
    self._q_init = 0.5
    self.new_sess()

  def new_sess(self):
    """Reset the agent for the beginning of a new session."""
    self._q = self._q_init * np.ones(self._n_actions)

  def get_choice(self) -> int:
    """Sample a choice, given the agent's current internal state."""
    choice_probs = 1 / (1 + np.exp(-(self._q[1] - self._q[0])))
    return np.random.choice(2, p=[choice_probs, 1 - choice_probs])

  def update(self,
             choice: int,
             reward: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    def softmax(choice):
        return np.exp(self._q[choice]) / np.sum(np.exp(self._q))
  
    opp_choice = 1 - choice
    self._q[choice] = (1. - self._alpha_f) * self._q[choice] + self._alpha_l * (reward - self.v) * (1. - softmax(choice))
    self._q[opp_choice] = (1. - self._alpha_f) * self._q[opp_choice] - self._alpha_l * (reward - self.v) * softmax(opp_choice)

    self.v = (1 - self._alpha_v) * self.v + self._alpha_v * reward

  @property
  def q(self):
    return np.hstack([[self._q[1] - self._q[0]], [self.v]])