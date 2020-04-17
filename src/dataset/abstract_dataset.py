from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np
from tqdm import tqdm
from abc import abstractmethod
from collections import defaultdict, OrderedDict

from torch.utils.data import Dataset

class AbstractDataset(Dataset):
    """
    All dataset loader classes will inherit from this class.
    """

    def __init__(self, config):
        """ Initialize data loader class, to be overwritten. """
        pass

    def __len__(self):
        return self.num_instances

    def __getitem__(self, idx):
        """ To be overwritten. """
        raise NotImplementedError("Method (generate_labels)")

    def get_samples(self, num_samples):
        """ Get smaple instances for visualization during training.
        Args:
            num_samples: number of samples; int
        """
        samples = []
        for i in range(num_samples):
            # randomly select sample index
            idx = np.random.randint(0, len(self)-1)
            sample = self.__getitem__(idx)
            samples.append(sample)

        return samples
        #return self.collate_fn(samples)

    def get_instance(self):
        """ Get a single instances for debugging.  """
        # randomly select sample index
        idx = np.random.randint(0, len(self)-1)
        return self.__getitem__(idx)

    def get_iteration_per_epoch(self):
        """ Get the number of iterations for each epoch given batch size """
        return self.num_instances / self.batch_size

    def _exist_data(self, paths):
        for k,v in paths.items():
            if not os.path.exists(v):
                return False
        return True

    def _build_vocab(self, anns, frequency_threshold=1):
        # count frequencies of words from captions
        frequency = defaultdict(lambda: 0)
        for qid,ann in tqdm(anns.items(), desc="Count frequency"):
            for w in ann["tokens"]:
                frequency[w] += 1

        # thresholding vocabulary
        cw = sorted([(cnt,w) for w,cnt in frequency.items()], reverse=True)
        words = sorted([w for w,cnt in frequency.items() if cnt >= frequency_threshold])
        print("Thresholding with {}: from {} -> {} ({:.3f})".format(
                frequency_threshold, len(frequency.keys()), len(words),
                len(words) / len(frequency.keys())))
        print("Top 20 words and their counts")
        print("\n".join(map(str, cw[:20])))

        # construct vocabulary
        wtoi = OrderedDict() # mapping word to index
        wtoi["<PAD>"], wtoi["<UNK>"] = 0, 1 # PAD, UNKNOWN tokens
        wtoi["<S>"], wtoi["<E>"]     = 2, 3 # START, END tokens
        for wi,w in enumerate(words):
            wtoi[w] = wi + 4 # words start in 4

        return wtoi

    def _encode_query(self, anns, wtoi, max_length=30):

        labels, lengths = {}, {}
        for qid,ann in tqdm(anns.items(), desc="Encoding query"):
            tokens = ann["tokens"]

            # obtain query labels and their lengths
            lengths[qid] = min(len(tokens), max_length)
            labels[qid] = np.zeros((max_length), dtype=np.int)

            # words -> labels
            for wi,w in enumerate(tokens):
                if wi == max_length: break
                labels[qid][wi] = wtoi.get(w, 1)   # 1: <UNK> token

        encoded = {
            "query_lengths": lengths,
            "query_labels": labels,
        }
        return encoded

    def get_fixed_length_feat(self, feat, num_segment, start_pos, end_pos):
        nfeats = feat[:,:].shape[0]
        if nfeats <= self.S:
            stride = 1
        else:
            stride = nfeats * 1.0 / num_segment
        if self.split != "train":
            spos = 0
        else:
            random_end = -0.5 + stride
            if random_end == np.floor(random_end):
                random_end = random_end - 1.0
            spos = np.random.random_integers(0,random_end)
        s = np.round( np.arange(spos, nfeats-0.5, stride) ).astype(int)
        start_pos =  float(nfeats-1.0) * start_pos
        end_pos = float(nfeats-1.0) * end_pos

        if not (nfeats < self.S and len(s) == nfeats) \
                and not (nfeats >= self.S and len(s) == num_segment):
            s = s[:num_segment] # ignore last one
        assert (nfeats < self.S and len(s) == nfeats) \
                or (nfeats >= self.S and len(s) == num_segment), \
                "{} != {} or {} != {}".format(len(s), nfeats, len(s), num_segment)

        start_index, end_index =  None, None
        for i in range(len(s)-1):
            if s[i] <= end_pos < s[i+1]:
                end_index = i
            if s[i] <= start_pos < s[i+1]:
                start_index = i

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = num_segment-1

        cur_feat = feat[s, :]
        nfeats = min(nfeats, num_segment)
        out = np.zeros((num_segment, cur_feat.shape[1]))
        out [:nfeats,:] = cur_feat
        return out, nfeats, start_index, end_index

    def collate_fn(self, data):
        batch = {key: [d[key] for d in data] for key in data[0]}
        if len(data) == 1:
            for k,v in batch.items():
                if k in self.tensor_items:
                    batch[k] = torch.cat(batch[k], 0)
                else:
                    batch[k] = batch[k][0]
        else:
            for key in self.tensor_items:
                batch[key] = torch.cat(batch[key], 0)

        return batch

    @abstractmethod
    def generate_labels(self, config):
        """ To be overwritten.
        Generate labels.
        """
        raise NotImplementedError("Method (generate_labels)")
