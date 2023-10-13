from genrlise.contexts.context_encoder import ContextEncoder
from genrlise.envs.base_env import BaseEnv
from genrlise.utils.types import Context


class ContextEncoderWrapperEnv(BaseEnv):
    """This is a context encoder wrapper
    """
    def __init__(self, base_env: BaseEnv, context_encoder: ContextEncoder) -> None:
        super().__init__([-10])
        self.original_env = base_env
        self.encoder = context_encoder
        
        self._good_keys = {
            'original_env', 'encoder', 'get_context',
            '__getattribute__', 
        }
        
    def __getattribute__(self, attr):
        # Pass through all other calls to the base env.
        dict = object.__getattribute__(self, '__dict__')
        goods = object.__getattribute__(self, '_good_keys')
        
        if attr in goods:
            return object.__getattribute__(self, attr)
    
        if 'original_env' not in dict:
            raise AttributeError
        og = object.__getattribute__(self, 'original_env')
        return getattr(og, attr)

    def get_context(self) -> Context:
        return self.encoder.get_context()