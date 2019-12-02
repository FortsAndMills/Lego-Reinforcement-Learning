'''
Both backbone and head are augmented with cache, which stores output for batch
inside the batch itself and reuses the cache if the "forward" pass is called again for the same batch.

In this modular framework cache is crucial as different modules may call the same heads
for the same batch without knowing about each other.
Consider Twin DQN with shared replay buffer as an example why the cache is needed.

The cache is implemented using the decorator "cached" for forward methods.

Same principle can be applied to calculation of intermediate tensors like Q-functions, policies, etc.
'''

def cached(forward):
    '''
    Decorator to augment "forward" pass with cache.
    '''
    def cached_forward(self, *input, **kwargs):
        # if storage is not provided, the cache is considered to be empty.
        # "cache_name" parameter can also be provided to distinguish passes
        # for state and for next state from the same batch.

        storage = kwargs.get("storage", None)
        cache_name = kwargs.get("cache_name", None)
        
        key = self.name if cache_name is None else self.name + " output for " + cache_name

        if storage is not None and key in storage:
            self.system.debug(self.name, f"reused {'' if cache_name is None else 'output for ' + cache_name} from cache!")
            return storage[key]
            
        output = forward(self, *input, **kwargs)
        
        self.system.debug(self.name, f"forward pass {'' if cache_name is None else 'for ' + cache_name} computed.")
        if storage is not None:
            storage[key] = output
        return output

    return cached_forward

def batch_cached(name):
    def batch_cached(f):
        def cached_function(self, batch):
            key = self.name + " " + name
            if key in batch:
                self.debug(f"{name} for the batch has already been calculated, used from cache!")
                return batch[key]       

            batch[key] = f(self, batch)
            return batch[key]
        return cached_function
    return batch_cached