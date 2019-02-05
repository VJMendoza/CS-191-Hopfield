from numpy import array
import PatternUtil

class Hopfield:

    def __init__(self):
        self.W = 0

    def train(self, patterns):
        """
        patterns is a numpy array holding the patterns to be learned
        """
        from numpy import zeros, outer, diag_indices
        row,col = patterns.shape
        self.W = zeros((col, col))

        for p in patterns:
            self.W = self.W + outer(p, p)
        
        self.W[diag_indices(col)] = 0
        self.W = self.W/row

    def recall(self, patterns, iter=5):
        from numpy import vectorize, dot
        sgn = vectorize(lambda x: -1 if x<0 else +1)
        for _ in range(iter):
            patterns = sgn(dot(patterns, self.W))
        
        return patterns



A = """
.XXX.
X...X
XXXXX
X...X
X...X
"""

A_c = """
.X.X.
..X.X
.XX.X
X.X..
.X..X
"""

B = """
XXXX.
X...X
XXXXX
X...X
XXXX.
"""

B_c = """
.X.X.
X.X..
.X..X
XX..X
..XX.
"""

patterns_to_be_learned = array([PatternUtil.to_nparray(A), PatternUtil.to_nparray(B)])
patterns_to_be_recalled = array([PatternUtil.to_nparray(A_c), PatternUtil.to_nparray(B_c)])

for p in patterns_to_be_recalled:
    PatternUtil.display_pattern(p)

hf1 = Hopfield()
hf1.train(patterns_to_be_learned)

recalled_pattern = hf1.recall(patterns_to_be_recalled, 10)

for p in recalled_pattern:
    PatternUtil.display_pattern(p)