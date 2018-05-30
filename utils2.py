from textblob import Word
import torch.nn as nn
import torch.nn.functional as F


def calculate_wordnet_distance(target_list):
    synsets_list = []
    distance_matrix = []
    for label_str in target_list:
        if label_str == 'aquarium_fish':
            label_str = 'fish'
        if label_str == 'maple_tree':
            label_str = 'maple'
        s_w_a = 0
        synsets = Word(label_str.lower()).synsets
        label_name = synsets[0]
        for label in synsets:
            if label_str == 'bear' or label_str == 'fish' or label_str == 'bed' or label_str == 'crab' or \
                            label_str == 'fox' or label_str == 'mouse' or label_str == 'rocket' or label_str == 'shark' \
                    or label_str == 'snake' or label_str == 'tank' or label_str == 'train' or label_str == 'lion' or \
                            label_str == 'man':
                label_name = synsets[0]
                break
            if label_str == 'beaver':
                label_name = synsets[6]
                break
            if label_str == 'whale':
                label_name = synsets[1]
                break
            if label_str == 'tiger':
                label_name = synsets[1]
                break
            if label_str == 'skunk':
                label_name = synsets[3]
                break
            if label._lexname == 'noun.animal':
                label_name = label
            # sim = label.wup_similarity(Word('animal').synsets[0])
            # if sim > s_w_a:
            #     label_name = label
            #     s_w_a = sim
        synsets_list.append(label_name)
    for label_name in synsets_list:
        label_distance = []
        for label_name2 in synsets_list:
            # label_distance.append((label_name.max_depth() - label_name.lowest_common_hypernyms(
            #     label_name2,
            #     simulate_root=True, use_min_depth=True
            # )[0].max_depth()))
            # label_distance.append((label_name.wup_similarity(label_name2)))
            if label_name.path_similarity(label_name2) != 1:
                label_distance.append(
                    (label_name.wup_similarity(label_name2) + label_name.path_similarity(label_name2)))
            else:
                label_distance.append((label_name.wup_similarity(label_name2)))
        distance_matrix.append(label_distance)
    return distance_matrix


class FineTuneModel(nn.Module):
    def __init__(self, original_model, num_classes):
        super(FineTuneModel, self).__init__()
        self.features = nn.Sequential(*list(original_model.modules())[:-1])
        self.classifier = nn.Sequential(nn.Linear(2048, num_classes))

        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


class FineTuneModel_alex(nn.Module):
    def __init__(self, original_model, num_classes):
        super(FineTuneModel_alex, self).__init__()
        self.features = original_model.features
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(256 * 6 * 6, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4096, num_classes))
        self.modelName = 'alexnet'

        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), 256 * 6 * 6)
        y = self.classifier(f)
        return y


class FineTuneModel_Dense(nn.Module):
    def __init__(self, original_model, num_classes):
        super(FineTuneModel_Dense, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(1920, num_classes))

        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = F.avg_pool2d(f, kernel_size=7).view(f.size(0), -1)
        y = self.classifier(f)
        return y
