from textblob import Word


def calculate_wordnet_distance(target_list):
    synsets_list = []
    distance_matrix = []
    for label in target_list:
        synsets_list.append(Word(label.lower()).synsets)
    for label_name in synsets_list:
        label_name = label_name[0]
        label_distance = []
        for label_name2 in synsets_list:
            label_name2 = label_name2[0]
            label_distance.append(label_name.wup_similarity(label_name2))
        distance_matrix.append(label_distance)
    return distance_matrix
