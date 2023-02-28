import torch
from transformers import BertModel, BertTokenizer


model_path = "/home/nfs_data/zhanggh/tmp/mrpc/"

tokenizer = BertTokenizer.from_pretrained(model_path)
device = torch.device('cuda')

# 载入模型
model = BertModel.from_pretrained(model_path).to(device)
# 输入文本
input_text = "Here is some text to encode"
# 通过tokenizer把文本变成 token_id
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
# input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
input_ids = torch.tensor([input_ids]).to(device)
# 获得BERT模型最后一个隐层结果

with torch.no_grad():
    outputs = model(input_ids, output_attentions=True)  # Models outputs are now tuples
    # hidden_states = outputs[2] # outputs["hidden_states"], given that return_dict = True
    attention_maps = outputs[2] # outputs["attentions"]
print(len(attention_maps))
print(attention_maps[0].shape)
torch.save(attention_maps, 'attentions.pt')