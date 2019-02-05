from numpy import array, arange, random, array_equal
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

    def async_recall(self, pattern, max_iters = 10): 
        order = arange(25)
        iter_count = 1 
        prev_iter = pattern.copy() 
        curr_iter = pattern.copy()
        while iter_count != max_iters:
            random.shuffle(order)
            for order_number in order:
                curr_iter[order_number] = self.update_node_async(curr_iter, order_number)
            if(array_equal(prev_iter, curr_iter)):
                break
            else:
                prev_iter = curr_iter.copy()
                iter_count += 1
        print("Recall converged at ", iter_count, " iterations")
        return curr_iter

    def update_node_async(self, curr_iter, order_number):
        from numpy import dot
        sign = lambda x: -1 if x < 0 else +1
        return sign(dot(curr_iter, self.W[:,[order_number]]))

    def predict_from_pattern(self, labels, training_data, pattern):
        predictions = []
        for data in training_data: 
            predictions.append(array_equal(pattern,data))
        print("Prediction: ", labels[predictions.index(True)])