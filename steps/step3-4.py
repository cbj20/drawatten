import torch
from transformers import BertModel, BertTokenizer
import pickle

model_path = "/home/nfs_data/zhanggh/tmp/mrpc/"

tokenizer = BertTokenizer.from_pretrained(model_path)
device = torch.device('cuda')
cpu = torch.device('cpu')

# 载入模型
model = BertModel.from_pretrained(model_path).to(device)
# 输入文本
with open("/home/nfs_data/zhanggh/drawatten/tmp/mrpc/input_ids", "rb") as fp:
    input_ids = pickle.load(fp)
input_ids = torch.tensor([input_ids]).squeeze(0).to(device)
# 获得BERT模型最后一个隐层结果
length = input_ids.shape[0]
print(input_ids.shape)
batch_size = 32
attention_maps = []
with torch.no_grad():
    for i in range(length // batch_size):
        tmp = input_ids[i * batch_size: min((i+1) * batch_size, length)]
        print(tmp.shape)
        outputs = model(tmp, output_attentions=True)  # Models outputs are now tuples
        print(outputs[0].shape)
        print(outputs[1].shape)
        print(outputs[2][0].shape)
        print(outputs[2][1].shape)
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
        attention_maps.append(outputs[2][0].to(cpu)) # outputs["attentions"]
        attention_maps.append(outputs[2][1].to(cpu))
print(len(attention_maps))
attention_maps = torch.tensor(attention_maps)
torch.save(attention_maps, 'attentions.pt')