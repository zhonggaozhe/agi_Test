import dashscope
import json
import os
from dashvector import Client, Doc, DashVectorCode
from dashscope import TextEmbedding
from dashscope import Generation

DASHSCOPE_API_KEY = 'sk-c59ae6fdd16e49e882b0adc0fbf08655'
DASHVECTOR_API_KEY = 'sk-6SlGAyeOn6hvEgdSWgPIAcXYTHGfl0D0CB40397CD11EE804982A52EEF0843'

'''
生成向量
 input (Union[str, List[str], io.IOBase]): The text input,
                can be a text or list of text or opened file object,
                if opened file object, will read all lines,
                one embedding per line.
'''


def generate_embeddings(news):
    rsp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1,
        input=news
    )
    embeddings = [record['embedding'] for record in rsp.output['embeddings']]
    return embeddings if isinstance(news, list) else embeddings[0]


def search_relevant_news(question):
    # 初始化 dashvector client
    client = Client(api_key=DASHVECTOR_API_KEY,
                    endpoint='dashvector.cn-hangzhou.aliyuncs.com')

    # 获取刚刚存入的集合
    collection = client.get('news_embedings')
    assert collection

    # 向量检索：指定 topk = 1
    vectors = generate_embeddings(question)
    print(vectors)
    rsp = collection.query(vectors, output_fields=['raw'],
                           topk=1)
    assert rsp
    return rsp.output[0].fields['raw']


def answer_question(question, context):
    prompt = f'''请基于```内的内容回答问题。"
	```
	{context}
	```
	我的问题是：{question}。
    '''

    print(f'prompt:\n 'f'{prompt}')
    rsp = Generation.call(model='qwen-turbo', prompt=prompt)
    return rsp.output.text


def create_collection(vector_client):
    # 创建集合：指定集合名称和向量维度, text_embedding_v1 模型产生的向量统一为 1536 维
    # 创建一个名称为quickstart、向量维度为4、
    # 向量数据类型为float（默认）、
    # 距离度量方式为dotproduct（内积）的Collection
    # 并预先定义三个Field，名称为name、weight、age，数据类型分别为str、float、int
    # timeout为-1 ,开启create接口异步模式
    ret = vector_client.create(
        name='yimm_embedings',
        dimension=1536,
        metric='cosine',
        dtype=float,
        fields_schema={'question': str, 'answer': str},
        timeout=-1
    )
    if ret.code == DashVectorCode.Success:
        print('create collection success!')
    else:
        raise ValueError(f"ErrorCode:{ret.code} Message:{ret.message}")


def load_yimm_data(vector_client):
    # 打开你的json文件
    with open('yimm/train_uejl.json', 'r') as f:
        data = json.load(f)
        # 现在，data就是一个Python字典，你可以通过键来获取数据
    collection = vector_client.get('yimm_embedings')
    id = 0
    # ids = [id + i for i, _ in enumerate(data)]
    for obj in data:
        print(obj)
        id += 1
        vectors = generate_embeddings(str(obj))
        # generate_embeddings(obj)
        # 写入 dashvector 构建索引
        # rsp = collection.upsert(
        #     [
        #         Doc(id=str(id), vector=vector, fields={"question": doc , "answer": })
        #         for id, vector, doc in zip(ids, vectors, news)
        #     ]
        # )
        rsp = collection.insert(
            Doc(
                id=str(id),
                vector=vectors,
                fields={
                    # 设置创建Collection时预定义的Fileds Value
                    'question': obj['instruction'], 'answer': obj['output']
                }
            )
        )
        assert rsp


def search_relevant_yimm(question):
    # 初始化 dashvector client
    client = Client(api_key=DASHVECTOR_API_KEY,
                    endpoint='dashvector.cn-hangzhou.aliyuncs.com')

    # 获取刚刚存入的集合
    collection = client.get('yimm_embedings')
    assert collection

    # 向量检索：指定 topk = 1
    vectors = generate_embeddings(question)
    print(vectors)
    rsp = collection.query(vectors, output_fields=['question', 'answer'],
                           topk=3)
    assert rsp
    return rsp.output[0].fields['answer']

if __name__ == '__main__':
    dashscope.api_key = DASHSCOPE_API_KEY

    # 初始化 dashvector client
    client = Client(api_key=DASHVECTOR_API_KEY, endpoint='dashvector.cn-hangzhou.aliyuncs.com')
    # 没有库则新建一个
    # create_collection(client)

    # 装载医美数据
    # {"code": -2011,
    #  "message": "DashVectorSDK UpsertDocRequest vector type(<class 'float'>) is invalid and must be in [List, numpy.ndarray]",
    #  "requests_id": ""}
    # load_yimm_data(client)


    # 查询测试
    question = '什么是光子嫩肤？'
    context = search_relevant_yimm(question)
    answer = answer_question(question, context)

    print(f'question: {question}\n' f'answer: {answer}')



