class Individual:
    def __init__(self, tree):
        self.tree = tree
        self.fitness = None

    def copy(self):
        clone = Individual(self.tree.copy())
        clone.fitness = self.fitness
        return clone