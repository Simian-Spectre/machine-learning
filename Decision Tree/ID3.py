import math

def array_from_csv(path_to_csv):
    data = []
    with open (path_to_csv, 'r') as csv:
        for line in csv:
            data.append(line.strip().split(','))
    return data
                           
CAR_TRAIN = array_from_csv('Decision Tree\\Car Data\\train.csv')
CAR_EVAL = array_from_csv('Decision Tree\\Car Data\\test.csv')
BANK_TRAIN = array_from_csv('Decision Tree\\Bank Data\\train.csv')
BANK_EVAL = array_from_csv('Decision Tree\\Bank Data\\test.csv')

def entropy(data, index_of_feature):
    unique_values = get_unique_values(data, index_of_feature)
    num_total_examples = len(data)

    entropy = 0
    for value in unique_values:
        num_examples_with_value = get_count(data, value, index_of_feature)
        ratio = num_examples_with_value / num_total_examples
        if ratio != 0:
            entropy -= ratio * math.log2(ratio)

    return entropy

def calculate_information_gain(data, index_of_feature, index_of_label):
    unique_values = get_unique_values(data, index_of_feature)
    num_total_examples = len(data)
    weighted_entropy = 0
  
    for value in unique_values:
        subset = []
        for example in data:
            if example[index_of_feature] == value:
                subset.append(example)

        proportion = len(subset) / num_total_examples
        weighted_entropy += proportion * entropy(subset, index_of_label)

    label_entropy = entropy(data, index_of_label)
    information_gain = label_entropy - weighted_entropy 
    return information_gain

def get_unique_values(data, index_of_feature):
    unique_values = []

    for example in data:
        value = example[index_of_feature]
        if value not in unique_values:
            unique_values.append(value)
    
    return unique_values

def get_count(data, value, index_of_feature):
    count = 0
    for example in data:
        if example[index_of_feature] == value:
            count += 1
    return count

def get_next_best_node_information_gain(data):
    best_information_gain = 0

    for i in range(len(data[0])-1):
        ent = entropy(data, i)
        information_gain = calculate_information_gain(data, i, 6)
        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_feature = i

    return best_feature

def id3(data, index_of_feature, features):
    if len(get_unique_values(data, index_of_feature)) == 1 or len(features) == 0:
        labels = []
        for example in data:
            labels.append(example[6])
        return max(set(labels), key=labels.count)    
    
    best_feature = get_next_best_node_information_gain(data)
    features = [f for f in features if f != best_feature]
    tree = {best_feature: {}}
    
    for value in get_unique_values(data, best_feature):
        subset = []
        for example in data:
            if example[index_of_feature] == value:
                subset.append(example)

        tree[best_feature][value] = id3(subset, index_of_feature, features)

    return tree 

CAR_MAPPER = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

def print_tree(tree, mapper, num_tabs):
    for key in tree:
        if isinstance(key, int):
            print("\t" * num_tabs + mapper[key])
            if not isinstance(tree, list):
                print_tree(tree[key], mapper, num_tabs + 1)
        else:
            print("\t" * num_tabs + key)

tree = id3(CAR_TRAIN, 0, [0, 1, 2, 3, 4, 5])
print_tree(tree, CAR_MAPPER, 0)