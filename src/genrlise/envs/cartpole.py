from genrlise.envs.base_cartpole_env import BaseCartPoleEnv

# A set of cartpole environments with various options.
class CartPoleContinuous(BaseCartPoleEnv):
    metadata = BaseCartPoleEnv.metadata
    def __init__(self, **kwargs):
        super().__init__(continuous_actions=True, **kwargs)


class CartPoleDiscrete(BaseCartPoleEnv):
    def __init__(self, **kwargs):
        super().__init__(continuous_actions=False, **kwargs)
        
        

class CartPoleContinuousInfiniteActions(BaseCartPoleEnv):
    def __init__(self, **kwargs):
        super().__init__(continuous_actions=True, infinite_actions=True, **kwargs)
        
        
class CartPoleContinuousLargeActions(BaseCartPoleEnv):
    def __init__(self, **kwargs):
        super().__init__(continuous_actions=True, large_actions=True, **kwargs)
        
        
class CartPoleContinuousArbitraryActions(BaseCartPoleEnv):
    def __init__(self, action_magnitude, **kwargs):
        super().__init__(continuous_actions=True, action_magnitude=action_magnitude, **kwargs)
