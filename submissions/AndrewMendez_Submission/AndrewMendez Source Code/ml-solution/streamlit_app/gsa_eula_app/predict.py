import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

import torch
import torch.nn as nn

class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea
def load_checkpoint(load_path, model,device):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']
def load_model_and_tokenizer(path_to_model, device):
    '''
    function to load pretrained model and tokenizer
    '''
    best_model = BERT().to(device)

    load_checkpoint(path_to_model + '/model.pt', best_model,device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return best_model,tokenizer
def predict(model,inputs):
    #print('model(inputs): ', model(inputs))
    return model.encoder(inputs)[0]
def custom_forward(model,inputs):
    preds = predict(model,inputs)
    return torch.softmax(preds, dim = 1)[:, 0] # for negative attribution, torch.softmax(preds, dim = 1)[:, 1] <- for positive 


def tokenize_text(text,tokenizer,device,ref_token_id,sep_token_id,cls_token_id):
    text_ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=128)
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    input_ids = torch.tensor([input_ids], device=device)
    return input_ids
def get_prediction_and_confidence(model,text,tokenizer,device,ref_token_id,sep_token_id,cls_token_id):
    '''
    pass text and return prediction label and confidence
    '''
    text_ids = tokenize_text(text,tokenizer,device,ref_token_id,sep_token_id,cls_token_id)
    score = predict(model,text_ids)
    pred_label = torch.argmax(score[0]).cpu().numpy()
    confidence = torch.softmax(score, dim = 1)[0][pred_label].cpu().detach().numpy()
    return pred_label,confidence