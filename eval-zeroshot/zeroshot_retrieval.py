import argparse
import torch
from tqdm import tqdm

from datasets.clip_dataset import load_data_retreival
from models.visual_encoder.backbone import load_ENLIGHT_enc

def get_clip_retrieval_args():
    parser = argparse.ArgumentParser('eval_CLIP', add_help=False)
    parser.add_argument('--database', default='./', type=str)
    parser.add_argument("--data", default='ut-0', type=str, choices=['ut-0', 'ut-1', 'ut-2', 'ut-3', 'ut-4', 'ut-5'], help='')
    parser.add_argument("--batch_size", default=512, type=int, help='')
    parser.add_argument("--num_workers", default=8, type=int, help='')
    parser.add_argument("--pretrained_path", required=True, type=str)
    parser.add_argument("--pretrained_source", default='openai', type=str, help='')
    parser.add_argument("--cache_dir", default='./cache', type=str, help='')
    return parser.parse_args()


def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def _get_image_embeddings(model, model_source, images):
    if model_source == 'openai':
        image_features = model.encode_image(images)
    else:
        raise ValueError(f'Model source: {model_source}')
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

def _get_text_embeddings(model, model_source, tokenize_func, texts, device):
    if model_source == 'openai':
        token_ids = tokenize_func(texts = texts).to(device)
        text_embeddings = model.encode_text(token_ids)
    else:
        raise ValueError(f'Model source: {model_source}')
  
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings

@torch.no_grad()
def retrieval_func(model, model_source, tokenize_func, dataloader, device, recall_k_list):
    n_gpu = torch.cuda.device_count()
    device = 'cuda' if n_gpu>0 else 'cpu'
    model = model.to(device)
    model.eval()

    texts = dataloader.dataset.cap_list
    text_embeddings = _get_text_embeddings(model, model_source, tokenize_func, texts, device)
    all_i2tscores = []
    all_i2tinds = []
    metrics = {}
    num_total = len(dataloader)
    for idx, (batch_images, batch_i2t_idx) in tqdm(enumerate(dataloader)):
        batch_images_emb = _get_image_embeddings(model, model_source, batch_images.to(device))
        scores = batch_images_emb @ text_embeddings.t()
        all_i2tscores.append(scores)
        all_i2tinds.append(batch_i2t_idx.to(device))
        print(f'[{idx+1}/{num_total}]')
    all_i2tinds = torch.cat(all_i2tinds)
    all_i2tscores = torch.cat(all_i2tscores) #(nimg, ncls)
    all_probs = all_i2tscores.softmax(dim=-1)
    all_preds = all_probs.argmax(dim=-1)
    num_cls = all_probs.shape[-1]

    # I2T retrieval
    batch_size = batch_images.shape[0]
    i2t_positive_pairs = torch.zeros_like(all_i2tscores, dtype=bool)
    i2t_positive_pairs[torch.arange(len(all_i2tscores)), all_i2tinds] = True
    
    for recall_k in [k for k in recall_k_list if k < num_cls]:
        metrics[f"I2T_R@{recall_k}"] = (batchify(recall_at_k, all_i2tscores, i2t_positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
    # T2I retrieval
    t2iscores = all_i2tscores.T
    t2i_positive_pairs = i2t_positive_pairs.T
    for recall_k in recall_k_list:
        metrics[f"T2I_R@{recall_k}"] = (batchify(recall_at_k, t2iscores, t2i_positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
    
    all_probs = all_probs.cpu().numpy()
    all_preds = all_preds.cpu().numpy()
    all_i2tinds = all_i2tinds.cpu().numpy()
    return metrics

if __name__ == "__main__":
    args = get_clip_retrieval_args()
    
    model, img_processor, tokenize_func = load_ENLIGHT_enc(args.pretrained_path,
                                                           args.cache_dir,
                                                           return_tokenizer=True)

    dataloader = load_data_retreival(img_processor,
                                    data=args.data, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers)
    metrics = retrieval_func(model,
                            model_source=args.pretrained_source,
                            tokenize_func=tokenize_func,
                            dataloader=dataloader,
                            recall_k_list=[1, 5, 10, 20])
    print(metrics)