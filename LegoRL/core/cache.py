from LegoRL.representations.representation import Which

'''
Transformations are augmented with cache, 
which stores output for storage inside the storage itself
and reuses the cache if the "forward" pass is called again for the same storage.

In this modular framework cache is crucial as different modules may call the same heads
for the same storage without knowing about each other.
Consider Twin DQN with shared replay buffer as an example why the cache is needed.

The cache is implemented using the following decorators.
'''

def cached_forward(forward):
    '''
    Decorator to augment "forward" passes through networks with cache.
    '''
    def cached(self, storage=None, which=Which.current):
        suffix = lambda which: '' if which is None else " output for " + which.name
        key = self.name + suffix(which)

        if storage is not None:
            if key in storage:
                self.debug(f"reused{suffix(which)} from cache!")
                return storage[key]
            elif self.name + suffix(Which.all) in storage:
                self.debug(f"reused{suffix(which)} from cache (used all)!")
                return storage[self.name + suffix(Which.all)].crop(which, storage.storage_type)
            elif which is Which.all and self.name + suffix(Which.current) in storage:
                self.debug(f"needs all, and current is in cache; will compute only for last")
                output_current = storage[self.name + suffix(Which.current)]
                output_last = cached(self, storage, Which.last)
                output = output_current.append(output_last)
                storage[key] = output
                return output

        output = forward(self, storage, which)
        
        self.debug(f"forward pass{suffix(which)} computed.")
        if storage is not None:
            storage[key] = output
        return output

    return cached

def storage_cached(name):
    '''
    Decorator fabric to augment some calculations with check if it was already computed.
    input: name - str, marker of computations.
    '''
    def cached(f):
        def cached_function(self, storage):
            key = self.name + " " + name
            if key in storage:
                self.debug(f"{name} for the batch has already been calculated, used from cache!")
                return storage[key]    

            storage[key] = f(self, storage)
            return storage[key]
        return cached_function
    return cached