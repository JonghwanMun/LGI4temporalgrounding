import random
import numpy as np

def compute_tiou(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return float(intersection) / union

def rank(pred, gt):
    for i,p in enumerate(pred):
        if gt[0] == p[0] and gt[1] == p[1]:
            return i+1
    return 9999999

def get_evaluator(dt="didemo"):
    if dt in ["anet", "charades"]:
        return TALLEvaluator()
    else:
        raise NotImplementedError("Not supported dataset type ({})".format(dt))

class TALLEvaluator(object):
    def __init__(self):
        self.tiou_threshold = [0.1, 0.3, 0.5, 0.7]
        self.metrics = ["R1-0.1", "R1-0.3", "R1-0.5", "R1-0.7", "mIoU"]
        #self.metrics = ["R1-0.3", "R1-0.5", "R1-0.7",
        #                "R5-0.3", "R5-0.5", "R5-0.7"]
        self.duration = None

    def get_metric(self):
        return "R1-0.5"

    def set_duration(self, duration=[]):
        if len(duration) == 0:
            self.duration = None
        else:
            self.duration = duration

    def eval_instance(self, pred, gt, topk):
        """ Compute Recall@topk at predefined tiou threshold for instance
        Args:
            pred: predictions of starting/end position; list of [start,end]
            gt: ground-truth of starting/end position; [start,end]
            topk: rank of predictions; int
        Return:
            correct: flag of correct at predefined tiou threshold [0.3,0.5,0.7]
        """
        correct = {str(tiou):0 for tiou in self.tiou_threshold}
        find = {str(tiou):False for tiou in self.tiou_threshold}
        if len(pred) == 0:
            return correct

        if len(pred) > topk:
            pred = pred[:topk]

        best_tiou = 0
        for loc in pred:
            cur_tiou = compute_tiou(loc, gt)

            if cur_tiou > best_tiou:
                best_tiou = cur_tiou

            for tiou in self.tiou_threshold:
                if (not find[str(tiou)]) and (cur_tiou >= tiou):
                    correct[str(tiou)] = 1
                    find[str(tiou)] = True

        return correct, best_tiou

    def eval(self, preds, gts):
        """ Compute R@1 and R@5 at predefined tiou threshold [0.3,0.5,0.7]
        Args:
            pred: predictions consisting of starting/end position; list
            gt: ground-truth of starting/end position; [start,end]
        Return:
            correct: flag of correct at predefined tiou threshold [0.3,0.5,0.7]
        """
        num_instances = float(len(preds))
        all_rank1 = {"R1-"+str(tiou):0 for tiou in self.tiou_threshold}
        all_rank5 = {"R5-"+str(tiou):0 for tiou in self.tiou_threshold}
        miou = 0

        ii = 0
        pt_idx = random.randint(0, len(gts)-1)
        for pred,gt in zip(preds, gts):
            if ii == pt_idx:
                if self.duration is not None:
                    print("pred: {}\tgt: {}\ttIoU: {:.4f}".format(
                        str(np.array(pred)/self.duration[ii]),
                        str(np.array(gt)/self.duration[ii]),
                        compute_tiou(np.array(pred).squeeze()/self.duration[ii],
                                   np.array(gt).squeeze()/self.duration[ii])
                    ))
                else:
                    print("pred: {}\tgt: {}\ttIoU: {}".format(
                            str(pred), str(gt), compute_tiou(np.array(pred).squeeze(), gt)))

            # compute rank1
            correct, _ = self.eval_instance(pred, gt, topk=1)
            for tiou in self.tiou_threshold:
                all_rank1["R1-"+str(tiou)] += correct[str(tiou)]

            # compute rank5
            correct, iou = self.eval_instance(pred, gt, topk=5)
            miou += iou
            for tiou in self.tiou_threshold:
                all_rank5["R5-"+str(tiou)] += correct[str(tiou)]

            ii += 1

        return all_rank1, all_rank5, miou
