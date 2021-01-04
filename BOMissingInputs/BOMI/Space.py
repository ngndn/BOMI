class Space:
    def __init__(self, lower, upper, inIsUniformSpaced=True):
        self.lower = lower
        self.upper = upper
        self.isUniformlySpaced = inIsUniformSpaced
        self.valueSet = []

    # If values are not uniformly spaced then we need to define
    def set_values_set(self, arr_values):
        self.valueSet = arr_values

    def get_ceil(self, val):
        result = val
        if val in self.valueSet:
            return val
        for x in self.valueSet:
            if x > val:
                result = x
                break
        return result

    def get_floor(self, val):
        result = val
        if val in self.valueSet:
            return val
        for x in self.valueSet:
            if x < val:
                result = x
            else:
                break
        return result
