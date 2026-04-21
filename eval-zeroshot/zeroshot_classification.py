import argparse
import torch
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score

from datasets.clip_dataset import load_data_clip, TEMPLATE
from models.visual_encoder.backbone import load_ENLIGHT_enc

def get_clipeval_args():
    parser = argparse.ArgumentParser('clipeval', add_help=False)
    parser.add_argument('--database', default='./', type=str)
    parser.add_argument("--data", default='SPIDER_colon', type=str, help='')
    parser.add_argument("--batch_size", default=196, type=int, help='')
    parser.add_argument("--num_workers", default=8, type=int, help='')
    parser.add_argument("--aggfirst", default=0, type=int, help='')
    parser.add_argument("--cache_dir", default='./cache', type=str, help='')
    parser.add_argument("--pretrained_path", required=True, type=str)
    parser.add_argument("--pretrained_source", default='openai', type=str, help='')
    parser.add_argument("--task", default='classification', type=str, choices=['classification', 'grading'])
    return parser.parse_args()


@torch.no_grad()
def _text_to_embeddings(tokenize_func, classname, templates, model, model_source, device):
    if isinstance(templates,str): 
        templates  = [templates]
    texts = [template.replace('CLASSNAME', classname) for template in templates]
    # token_ids = tokenize(tokenizer, texts) # Tokenize with custom tokenizer
    if model_source == 'openai':
        token_ids = tokenize_func(texts = texts).to(device)
        text_embeddings = model.encode_text(token_ids)
    else:
        raise ValueError(f'Model source: {model_source}')
    # text_embeddings: [num_templates, embedding_dim]
    return text_embeddings

def _prompts_to_text_feats(model, model_source, classnames, templates, tokenize_func, device):
    """
    classnames: list of lists of classnames (one list of classnames per class)
    templates: list of templates 
    """
    text_feats = []
    for classnames_for_class in classnames:
        embeddings_for_class = []
        for classname in classnames_for_class:
            if isinstance(classname, str):
                class_embeddings = _text_to_embeddings(tokenize_func, classname, templates, model, model_source, device)
            else:
                class_embeddings = []
                for sub_classname in classname:
                    subclass_embeddings = _text_to_embeddings(tokenize_func, sub_classname, templates, model, model_source, device)
                    class_embeddings.append(subclass_embeddings)
            # embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1))
            embeddings_for_class.append(class_embeddings)
            # class_embedding: [num_classnames, num_templates, embedding_dim]
        text_feats.append(embeddings_for_class)
    return text_feats # [num_class, num_classnames, num_templates, embedding_dim]



def _similarity_and_agg(image_features, text_features):
    # image_features: [batch_size, embedding_dim]
    # text_features: List [num_class, num_classnames, num_templates, embedding_dim]
    out_sim = []
    for text_feat_cls in text_features:
        sim_cls = []
        for text_feat_clsname in text_feat_cls:
            if isinstance(text_feat_clsname, torch.Tensor): # [num_templates, embedding_dim]
                text_feat_clsname /= text_feat_clsname.norm(dim=-1, keepdim=True)
                sim = image_features @ text_feat_clsname.T #[bs, num_templates]
                sim = sim.mean(dim=1)
            else: # [num_subclsname, num_templates, embedding_dim]
                text_feat_clsname = torch.stack(text_feat_clsname).permute(2, 0, 1) # [embedding_dim, num_subclsname, num_templates]
                dim_subcls, dim_templates = text_feat_clsname.shape[1:]
                sim = (image_features @text_feat_clsname.reshape(-1, dim_subcls*dim_templates)) # [bs, embedding_dim] @ [embedding_dim, num_subclsname, num_templates] -> [bs, num_subclsname* num_templates]
                sim = sim.reshape(-1, dim_subcls, dim_templates) # [bs, num_subclsname, num_templates]
                sim = torch.max(sim.mean(dim=-1), dim=-1).values
            sim_cls.append(sim)
        sim_cls = torch.stack(sim_cls, dim=0) # [num_clsname, batch_size]
        sim_cls = sim_cls.mean(dim=0) # [batch_size]
        out_sim.append(sim_cls)
    return torch.stack(out_sim, dim=-1) # [batch_size, num_classes]


def _aggregate_text_feats(image_features, text_features):
    # image_features: [batch_size, embedding_dim]
    # text_features: List [num_class, num_classnames, num_templates, embedding_dim]
    out_feats = []
    bs = image_features.shape[0]
    for text_feat_cls in text_features:
        agg_text_feat_cls = []
        for text_feat_clsname in text_feat_cls:
            assert isinstance(text_feat_clsname, torch.Tensor), text_feat_clsname 
            # [num_templates, embedding_dim] -> [bs, embedding_dim]
            text_feat_clsname = text_feat_clsname.mean(dim=0)
            text_feat_clsname = text_feat_clsname[None].repeat(bs, 1)
    
            agg_text_feat_cls.append(text_feat_clsname)

        agg_text_feat_cls = torch.stack(agg_text_feat_cls, dim=1) # [bs, num_prompts_percls, embedding_dim]
        agg_text_feat_cls = agg_text_feat_cls.mean(dim=1) # [bs, embedding_dim]

        out_feats.append(agg_text_feat_cls)

    out_feats = torch.stack(out_feats, dim=-1) # [bs, embedding_dim, num_classes]
    return out_feats

def _cosine_similarity(image_features, text_features, aggfirst=True):
    if aggfirst: # agg->norm->prob
        text_features = _aggregate_text_feats(image_features, text_features)
        # [bs, embedding_dim, num_classes]
        text_features /= text_features.norm(dim=1, keepdim=True)
        sim = image_features[:,None] @ text_features #(BS, 1, D) @ (BS, D, num_cls) -> (BS, 1, num_cls)
        sim = sim.squeeze(1)
    else: # norm->prob->agg
        sim = _similarity_and_agg(image_features, text_features)
    return sim

@torch.no_grad()
def classify_func(model, model_source, dataloader, label_2_classname, templates, tokenize_func):
    n_gpu = torch.cuda.device_count()
    device = 'cuda' if n_gpu>0 else 'cpu'
    model = model.to(device)
    model.eval()
    
    classnames = [v if isinstance(v, list) else [v] for v in label_2_classname.values()]
    
    text_features = _prompts_to_text_feats(model = model,
                                          model_source = model_source,
                                          classnames = classnames,
                                          templates = templates,
                                          tokenize_func = tokenize_func,
                                          device = device)
    probs, gts, preds = [], [], []
    for images, labels, _ in tqdm.tqdm(dataloader):
        images = images.to(device)

        if model_source == 'openai':
            image_features = model.encode_image(images)
        else:
            raise ValueError(f'Model source: {model_source}')
            
        image_features = model(
                        image=images, 
                        out_norm=True,
                        with_head=True  # head must be used for zero-shot task
                        )[0]
        
        image_features /= image_features.norm(dim=-1, keepdim=True)

        mul_similarity = _cosine_similarity(image_features, text_features)
        
        prob = (model.logit_scale.exp().item() * mul_similarity).softmax(dim=-1) #(BS, num_cls)
        #model.logit_scale or model.logit_scale.exp() or 100
        pred = prob.argmax(dim=-1)

        probs.append(prob.cpu().numpy())
        preds.append(pred.cpu().numpy())
        gts.append(labels.cpu().numpy())

    
    probs = np.concatenate(probs, axis=0)
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    
    
    return gts, probs, preds


if __name__ == "__main__":
    args = get_clipeval_args()

    model, img_processor, tokenize_func = load_ENLIGHT_enc(args.pretrained_path,
                                                           args.cache_dir,
                                                           return_tokenizer=True)

    dataloader, label_2_prompt  = load_data_clip(database = args.database,
                                        dataname = args.data,
                                        img_processor = img_processor,
                                        batch_size = args.batch_size,
                                        num_workers = args.num_workers)
    

    gts, probs, preds = classify_func(model = model,
            model_source = args.pretrained_source,
            dataloader = dataloader,
            label_2_classname = label_2_prompt,
            templates = TEMPLATE, 
            tokenize_func=tokenize_func)
    
    if args.task == 'classification':
        n_classes = probs.shape[1]
        if n_classes == 2:
            class_probs = probs[:,1]
            macro_roc_kwargs = {}
        else:
            class_probs = probs
            macro_roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}
            #  Stands for One-vs-one. 
            # Computes the average AUC of all possible pairwise combinations of classes. 
            # Insensitive to class imbalance when average == 'macro'      
        macro_auc = roc_auc_score(gts, class_probs, **macro_roc_kwargs)
    elif args.task == 'grading':
        linear_kappa = cohen_kappa_score(gts, preds, weights='linear')