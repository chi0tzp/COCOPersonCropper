import torch


def detection_collate(batch):
    person_imgs = []
    keypoint_lists = []
    for sample in batch:
        if sample[0].size() != torch.Size([0]):
            person_imgs.append(sample[0])
        if len(sample[1]) != 0:
            keypoint_lists += sample[1]
        # targets.append(torch.FloatTensor(sample[1]))

    return torch.cat(person_imgs, 0), keypoint_lists
