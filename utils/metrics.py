import numpy as np
from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, 
                             classification_report, roc_auc_score)

class EvalMeterRercords(object):
    """Record evalation metric across steps, return best, early stop if setting criteria"""
    def __init__(self, early_stop_times=0):
        self.reset()
        self.early_stop_times = early_stop_times

    def reset(self):
        self.history = {}
        self.best_metric = -np.inf
        self.best_iter = 0
        self.best_update_buffer = []

    def if_early_stop(self,):
        if self.early_stop_times>0:
            if len(self.best_update_buffer)<self.early_stop_times:
                return False
            else:
                recent_best_update = self.best_update_buffer[-self.early_stop_times:]
                return np.all(np.array(recent_best_update)==False) # if all False
        else:
            return False

    def update(self, metric, nstep):
        self.history.update({nstep: metric})
        update_best = metric>self.best_metric
        self.best_update_buffer += [update_best]
        if update_best:
            self.best_iter = nstep
            self.best_metric = metric
        return update_best


def classification_metrics(probs, gts, auc_metric='auc',f1_metric='f1'):
    """
    probs: (n_samples, n_classes)
    gts: (n_samples,)
    """
    probs = np.array(probs)
    gts = np.array(gts)
    preds = np.argmax(probs, axis=1)
    
    metric_dict = {}
    # Acc
    bacc = balanced_accuracy_score(gts, preds)
    metric_dict.update({'bacc': bacc})
    
    # F1
    cls_rep = classification_report(gts, preds, output_dict=True, zero_division=0)
    if f1_metric=='f1':
        F1 = cls_rep['macro avg']['f1-score']
        metric_dict.update({'f1':F1})
    elif f1_metric=='wf1':
        wF1  = cls_rep['weighted avg']['f1-score']
        metric_dict.update({'wf1': wF1})
    elif f1_metric=='kappa':
        quad_kappa = cohen_kappa_score(gts, preds, weights='quadratic')
        linear_kappa = cohen_kappa_score(gts, preds, weights='linear')
        metric_dict.update({'qKAPPA': quad_kappa, 'lKAPPA': linear_kappa})
    
    # AUC
    n_classes = probs.shape[1]
    if n_classes == 2:
        class_probs = probs[:,1]
        macro_roc_kwargs, wroc_kwargs = {}, {}
    else:
        class_probs = probs
        macro_roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}
        wroc_kwargs = {'multi_class': 'ovo', 'average': 'weighted'}
        #  Stands for One-vs-one. 
        # Computes the average AUC of all possible pairwise combinations of classes. 
        # Insensitive to class imbalance when average == 'macro'      
    if auc_metric == 'auc':
        metric_dict.update({'auc':roc_auc_score(gts, class_probs, **macro_roc_kwargs)})
    elif auc_metric == 'wauc':
        metric_dict.update({'wauc':roc_auc_score(gts, class_probs, **wroc_kwargs)})
    return metric_dict


def ensemble_5feval(datalist, isinternal=True):
    
    assert len(datalist) == 5
    kwargs = {'auc_metric':'auc', 'f1_metric':''}
    # Average of metrics
    proball, gtall = [],[]
    metric_f5 = defaultdict(list)
    for data in datalist:
        probs, gts = data['PROB'], data['GT']
        metricfold = classification_metrics(probs, gts, **kwargs)
        for key in ['bacc','auc']:
            metric_f5[key].append(metricfold[key])
        proball += [probs]
        gtall += [gts]

    metrics = {f'{key}_mean':sum(vlist)/len(vlist) for key, vlist in metric_f5.items()}
    if isinternal:
        # Metrics of concate predictions 
        proball = np.concatenate(proball)
        gtall = np.concatenate(gtall)
        metrics.update(classification_metrics(proball, gtall,**kwargs))
    else: #TODO: improve ensemble
        # Metrics of ensembled predictions
        proball = np.stack(proball, axis=0) #(5,N,2)
        gtall = np.stack(gtall, axis=0) #(5,N)
        gt = gtall[0]
        assert all(np.all(gt==gtall, axis=0))
        prob = np.mean(proball, axis=0) #(N,2)
        metrics.update(classification_metrics(prob, gt, **kwargs))
    metrics = {k:metrics[k] for k in sorted(metrics.keys())}
    return metrics


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

   
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))