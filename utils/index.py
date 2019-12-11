class Index(object):
    def __init__(self, value=0):
        self.value = value

    def __index__(self):
        return self.value

    def __iadd__(self, other):
        self.value += other
        return self

    def __add__(self, other):
        if isinstance(other, Index):
            return Index(self.value + other.value)
        elif isinstance(other, int):
            return Index(self.value + other)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, Index):
            return Index(self.value - other.value)
        elif isinstance(other, int):
            return Index(self.value - other)
        else:
            raise NotImplementedError

    def __eq__(self, other):
        return self.value == other

    def __repr__(self):
        return "Index({})".format(self.value)

    def __str__(self):
        return "{}".format(self.value)

    def set(self, other):
        if isinstance(other, int):
            self.value = other
        elif isinstance(other, Index):
            self.value = other.value
        else:
            raise NotImplementedError
