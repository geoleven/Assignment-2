


class NestedCV:
    def __init__(self, estimators, paramSpaces, repeats=10, outerLoopsN=5, innerLoopsN=3, seed=42):
        self.estimators = estimators
        self.paramSpaces = paramSpaces
        self.repeats = repeats
        self.outerLoopsN = outerLoopsN
        self.innerLoopsN = innerLoopsN
        self.seed = seed

    
