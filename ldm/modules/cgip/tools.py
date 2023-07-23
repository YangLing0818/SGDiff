import torch
import torch.nn.functional as F
import copy

def create_tensor_by_assign_samples_to_img(samples, sample_to_img, max_sample_per_img, batch_size):
    dtype, device = samples.dtype, samples.device
    N = batch_size
    D = samples.shape[1]
    assert (sample_to_img.max() + 1) == N

    samples_per_img = []
    for i in range(N):
        s_idxs = (sample_to_img == i).nonzero().view(-1)
        sub_sample = samples[s_idxs]
        len_cur = sub_sample.shape[0]
        if len_cur > max_sample_per_img:
            sub_sample = sub_sample[:max_sample_per_img, :]
        if len_cur < max_sample_per_img:
            zero_vector = torch.zeros([1, D]).to(device)
            padding_vectors = torch.cat([copy.deepcopy(zero_vector) for _ in range(max_sample_per_img - len_cur)], dim=0) # [res, D]
            sub_sample = torch.cat([sub_sample, padding_vectors], dim=0)
        sub_sample = sub_sample.unsqueeze(0)
        samples_per_img.append(sub_sample)
    samples_per_img = torch.cat(samples_per_img, dim=0).to(device)

    return samples_per_img

def idx_to_one_hot(idx, num_classes):
    result = F.one_hot(idx, num_classes)
    result = result.float()
    return result


def sample_json(vocab, scene_graphs):
    objs, triples, obj_to_img, triple_to_img = encode_scene_graphs(vocab, scene_graphs)
    return objs, triples, obj_to_img, triple_to_img


def encode_scene_graphs(vocab, scene_graphs):
    if isinstance(scene_graphs, dict):
        scene_graphs = [scene_graphs]

    objs, triples, obj_to_img = [], [], []
    obj_offset = 0
    for i, sg in enumerate(scene_graphs):
        sg['objects'].append('__image__')
        image_idx = len(sg['objects']) - 1
        for j in range(image_idx):
            sg['relationships'].append([j, '__in_image__', image_idx])

        for obj in sg['objects']:
            obj_idx = vocab['object_name_to_idx'].get(obj, None)
            if obj_idx is None:
                raise ValueError('Object "%s" not in vocab' % obj)
            objs.append(obj_idx)
            obj_to_img.append(i)
        for s, p, o in sg['relationships']:
            pred_idx = vocab['pred_name_to_idx'].get(p, None)
            if pred_idx is None:
                raise ValueError('Relationship "%s" not in vocab' % p)
            triples.append([s + obj_offset, pred_idx, o + obj_offset])
        obj_offset += len(sg['objects'])
    objs = torch.tensor(objs, dtype=torch.int64)
    triples = torch.tensor(triples, dtype=torch.int64)
    obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64)

    T = triples.shape[0]
    triple_to_img = torch.zeros([T, ], dtype=torch.int64)
    return objs, triples, obj_to_img, triple_to_img