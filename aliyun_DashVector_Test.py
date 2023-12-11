import os

import dashscope
from dashscope import TextEmbedding

from dashvector import Client, Doc


def prepare_data(path, batch_size=25):
    batch_docs = []
    for file in os.listdir(path):
        with open(path + '/' + file, 'r', encoding='utf-8') as f:
            batch_docs.append(f.read())
            if len(batch_docs) == batch_size:
                yield batch_docs
                batch_docs = []

    if batch_docs:
        yield batch_docs


def generate_embeddings(news):
    rsp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1,
        input=news
    )
    embeddings = [record['embedding'] for record in rsp.output['embeddings']]
    return embeddings if isinstance(news, list) else embeddings[0]


DASHSCOPE_API_KEY = 'sk-c59ae6fdd16e49e882b0adc0fbf08655'
DASHVECTOR_API_KEY = 'sk-6SlGAyeOn6hvEgdSWgPIAcXYTHGfl0D0CB40397CD11EE804982A52EEF0843'

if __name__ == '__main__':
    dashscope.api_key = DASHSCOPE_API_KEY

    # 初始化 dashvector client
    client = Client(api_key=DASHVECTOR_API_KEY, endpoint='dashvector.cn-hangzhou.aliyuncs.com')

    # # 创建集合：指定集合名称和向量维度, text_embedding_v1 模型产生的向量统一为 1536 维
    # rsp = client.create('news_embedings', 1536)
    # assert rsp

    # 加载语料
    id = 0
    collection = client.get('news_embedings')
    for news in list(prepare_data('CEC-Corpus/raw corpus/allSourceText')):
        ids = [id + i for i, _ in enumerate(news)]
        id += len(news)

        vectors = generate_embeddings(news)
        for id, vector, doc in zip(ids, vectors, news):
            print(f'id:{id}  vector:{vector}  doc:{doc}')
        # 写入 dashvector 构建索引
        # rsp = collection.upsert(
        #     [
        #         Doc(id=str(id), vector=vector, fields={"raw": doc})
        #         for id, vector, doc in zip(ids, vectors, news)
        #     ]
        # )
        # assert rsp