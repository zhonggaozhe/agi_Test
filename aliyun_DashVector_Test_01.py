import dashscope
from dashvector import Client
from dashscope import TextEmbedding
from dashscope import Generation


def generate_embeddings(news):
    rsp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1,
        input=news
    )
    embeddings = [record['embedding'] for record in rsp.output['embeddings']]
    return embeddings if isinstance(news, list) else embeddings[0]


def search_relevant_news(question):
    # 初始化 dashvector client
    client = Client(api_key='sk-6SlGAyeOn6hvEgdSWgPIAcXYTHGfl0D0CB40397CD11EE804982A52EEF0843',
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


if __name__ == '__main__':
    dashscope.api_key = 'sk-c59ae6fdd16e49e882b0adc0fbf08655'

    question = '海南安定追尾事故，发生在哪里？原因是什么？人员伤亡情况如何？'
    context = search_relevant_news(question)
    answer = answer_question(question, context)

    print(f'question: {question}\n' f'answer: {answer}')
