import stable_baselines3


class MySB3Monitor(stable_baselines3.common.monitor.Monitor):
    # a better sb3 monitor that does not complain about private attributes.
    def __getattr__(self, name):
        return getattr(self.env, name)
