from Hopfield import Hopfield
import InputUtil 
import PatternUtil

hf = Hopfield()
labels, training_patterns = InputUtil.load_training_data()
hf.train(training_patterns)

test_patterns = InputUtil.load_test_data()
for test in test_patterns: 
    print("Initial pattern:")
    PatternUtil.display_pattern(test)
    updated_pattern = hf.async_recall(test)
    print("Updated pattern:")
    PatternUtil.display_pattern(updated_pattern)
    hf.predict_from_pattern(labels,training_patterns,updated_pattern)
