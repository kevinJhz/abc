import json

import oneflow as flow
from flowvision import transforms
from flowvision import datasets
import oneflow as flow
from oneflow.utils.data import Dataset
from libai.data.structures import DistTensorData, Instance
from libai.tokenizer import BertTokenizer


class ExtractionDataSet(Dataset):
    def __init__(self, data_path, vocab_path, is_train, indc=None):
        if is_train:
            self.train = True
            data_path = data_path + "/train.json"
        else:
            data_path = data_path + "/dev.json"
        print("data_path is " + data_path)
        print("vocab_path is " + vocab_path)
        data = load_data(data_path=data_path, vocab_path=vocab_path, indc=indc)
        text = [t.numpy() for t in data['text']]
        mask = [t.numpy() for t in data['mask']]
        label = [t for t in data['label']]

        self.text = flow.tensor(text)
        self.mask = flow.tensor(mask)
        self.label = flow.tensor(label)

        print("--data.shape--")
        print(self.text.shape)
        print(self.mask.shape)
        print(self.label.shape)

    def __getitem__(self, idx):
        sample = Instance(
            input_ids=DistTensorData(
                flow.tensor(self.text[idx], dtype=flow.int64)
            ),
            attention_mask=DistTensorData(
                flow.tensor(self.mask[idx], dtype=flow.int64),
                placement_idx=-1
            ),
            tokentype_ids=DistTensorData(
                flow.tensor(self.label[idx], dtype=flow.int64),
                placement_idx=-1
            )

        )

        return sample

    def __len__(self):
        return len(self.text)


def map_id_rel():
    id2rel = {0: 'UNK', 1: '主演', 2: '歌手', 3: '简称', 4: '总部地点', 5: '导演', 6: '出生地', 7: '目', 8: '出生日期', 9: '占地面积',
              10: '上映时间', 11: '出版社', 12: '作者', 13: '号', 14: '父亲', 15: '毕业院校', 16: '成立日期', 17: '改编自', 18: '主持人',
              19: '所属专辑', 20: '连载网站', 21: '作词', 22: '作曲', 23: '创始人', 24: '丈夫', 25: '妻子', 26: '朝代', 27: '民族', 28: '国籍',
              29: '身高', 30: '出品公司', 31: '母亲', 32: '编剧', 33: '首都', 34: '面积', 35: '祖籍', 36: '嘉宾', 37: '字', 38: '海拔',
              39: '注册资本', 40: '制片人', 41: '董事长', 42: '所在城市', 43: '气候', 44: '人口数量', 45: '邮政编码', 46: '主角', 47: '官方语言',
              48: '修业年限'}
    rel2id = {}
    for i in id2rel:
        rel2id[id2rel[i]] = i
    return rel2id, id2rel


def load_data(data_path="train.json", vocab_path='./bert-base-chinese', indc=None):
    rel2id, id2rel = map_id_rel()
    max_length = 128
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    data = {}
    data['label'] = []
    data['mask'] = []
    data['text'] = []

    with open(data_path, 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()
        if indc is not None:
            temp = temp[:indc]
        for line in temp:
            dic = json.loads(line)
            if dic['rel'] not in rel2id:
                data['label'].append(0)
            else:
                data['label'].append(rel2id[dic['rel']])

            sent = dic['ent1'] + dic['ent2'] + dic['text']
            indexed_tokens = tokenizer.encode(sent)

            avai_len = len(indexed_tokens)
            while len(indexed_tokens) < max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = flow.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

            # Attention mask
            att_mask = flow.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            data['text'].append(indexed_tokens)
            data['mask'].append(att_mask)
    return data


class MnistDataSet(Dataset):
    def __init__(self, path, is_train, indc=None):
        print(path)
        self.data = datasets.MNIST(
            root=path,
            train=is_train,
            transform=transforms.ToTensor(),
            download=False
        )
        if indc is not None:
            self.data = flow.utils.data.Subset(dataset=self.data, indices=range(indc))

    def __getitem__(self, idx):
        sample = Instance(
            inputs=DistTensorData(
                flow.tensor(self.data[idx][0], dtype=flow.float32)
            ),
            labels=DistTensorData(
                flow.tensor(self.data[idx][1], dtype=flow.int), placement_idx=-1)
        )

        return sample

    def __len__(self):
        return len(self.data)
