import pandas as pd
import numpy as np
import xlsxwriter
import base64
from io import BytesIO
import asyncio
import aiohttp
import json
import streamlit as st

# Função para transformar df em excel
def to_excel(df):
	output = BytesIO()
	writer = pd.ExcelWriter(output, engine='xlsxwriter')
	df.to_excel(writer, sheet_name='Planilha1',index=False)
	writer.close()
	processed_data = output.getvalue()
	return processed_data
	
# Função para gerar link de download
def get_table_download_link(df):
	val = to_excel(df)
	b64 = base64.b64encode(val)
	return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download</a>'

def replace_column_with_review(df_reviews, list_string):
    for column in df_reviews.columns:
        if column in list_string:
            df_reviews.rename(columns={column: "Review"}, inplace=True)
    return df_reviews

# Função que recebe o dataframe de reviews e adiciona a string 'Comentário: ', retornando uma lista dos reviews
def make_reviews(df_reviews):
    list_reviews = []
    for i in list(df_reviews['Review']):
        review = "Comentário: " + i
        list_reviews.append(review)
        
    return list_reviews

# Função para criar lotes de reviews
def coletar_lotes(lista, tamanho_lote):
    lotes = [lista[i:i + tamanho_lote] for i in range(0, len(lista), tamanho_lote)]
    return lotes

# Função que cria contexto para o modelo com as subcategorizações e detalhamentos
def create_system(df_classes):
    
    list_sub = list(df_classes['Subcategoria'].dropna())
    list_detail = list(df_classes['Detalhamento'].dropna())

    string_sub = ', '.join(f'"{s}"' for s in list_sub)
    string_detail = ', '.join(f'"{s}"' for s in list_detail)
    
    system = f"""Haja como um classificador de texto. Irei fornecer um texto de um comentário de uma loja de aplicativos e 
    seu objetivo será classificar o comentário em 4 grupos de classes pré-estabelecidas que eu também vou fornecer.
    Para cada comentário, selecione um item de cada lista, respeitando exatamente o texto do item, sem qualquer variação.
    Lembre-se de utilizar apenas os itens que estão entre parênteses de seu respectivo grupo.

    Sua resposta deve conter exclusivamente:
    \n"Sentimento": "Positivo", "Negativo", "Neutro", "Misto"
    \n"Categoria": "Elogio", "Reclamação", "Sugestão", "Dúvida", "Indefinido"
    \n"Subcategoria": {string_sub}
    \n"Detalhamento": {string_detail}"""
    
    return system

async def get_data(session, body_mensagem):
    
    API_KEY = st.secrets["TOKEN_API"]

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    url = "/v1/chat/completions"
    
    response = await session.post(url, headers=headers, data=body_mensagem)
    body = await response.json()
    response.close()
    return body

# chatGPT - criação de respostas
async def get_chatgpt_responses(system, lotes_reviews):
    
    url_base = "https://api.openai.com"
    id_modelo = "gpt-3.5-turbo"
    
    session = aiohttp.ClientSession(url_base)
    tasks = []
    for review in lotes_reviews:
        
        review_string = '\n'.join(review)
        
        body_mensagem = {

            "model": id_modelo,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": '''Sua resposta deve ser apenas as classificações geradas de cada
                comentário dentro de um array, nada mais, no seguinte formato de exemplo: "['Sentimento', 'Categoria', 'Subcategoria', 'Detalhamento']"''' + review_string}
            ],
            "max_tokens":500,
            "temperature": 0.2
        }

        body_mensagem = json.dumps(body_mensagem)
        tasks.append(get_data(session,body_mensagem))

    data = await asyncio.gather(*tasks)

    await session.close()

    return data

# Normalização de resultados recebidos pela API
def normalize_results(results):
    df_results = pd.DataFrame(results)
    df_replies = pd.json_normalize(pd.DataFrame(df_results.explode('choices')['choices'])['choices'])

    return df_replies

# Tratamento de lotes de classificação
def clean_results(df_results):
    
    df_results['message.content'] = df_results['message.content'].str.replace("\n", ',')
    df_results['message.content'] = df_results['message.content'].apply(lambda x: eval('[' + x + ']'))
    df_results = df_results.explode('message.content').reset_index(drop=True)
    
    return df_results

# Acrescentar classificações no df de reviews, renomear colunas, adicionar valor Genérico caso não venha classificação da API
def format_results(df_reviews, df_results):
    
    df_reviews['results'] = df_results['message.content']
    df_reviews = pd.concat([df_reviews.drop('results', axis=1), df_reviews['results'].apply(pd.Series)], axis=1)
    df_reviews = df_reviews.rename(columns={0: 'Sentiment_pred', 1: 'Category_pred', 2: 'Subcategory_pred', 3: 'Detailing_pred'})

    df_reviews['Subcategory_pred'] = df_reviews['Subcategory_pred'].fillna('Genérico')
    df_reviews['Detailing_pred'] = df_reviews['Detailing_pred'].fillna('Genérico')
    df_reviews['Sentiment_pred'] = df_reviews['Sentiment_pred'].fillna('Genérico')
    df_reviews['Category_pred'] = df_reviews['Category_pred'].fillna('Genérico')
    
    df_reviews = df_reviews[['Review', 'Sentiment_pred', 'Category_pred', 'Subcategory_pred', 'Detailing_pred']]
    
    return df_reviews

# Substituir classificações que não estão na lista por nan
def replace_errors_with_nan(df_reviews, df_classes):

    # Padronizando strings de subcategorias e detalhamentos
    df_classes['Detalhamento'] = df_classes['Detalhamento'].str.capitalize()
    df_classes['Subcategoria'] = df_classes['Subcategoria'].str.capitalize()

    df_reviews['Detailing_pred'] = df_reviews['Detailing_pred'].str.capitalize()
    df_reviews['Subcategory_pred'] = df_reviews['Subcategory_pred'].str.capitalize()

    # Verificar valores da coluna "valor_a" com a coluna "lista_a"
    df_reviews['Subcategory_pred'] = np.where(df_reviews['Subcategory_pred'].isin(df_classes['Subcategoria']), df_reviews['Subcategory_pred'], np.nan)

    # Verificar valores da coluna "valor_b" com a coluna "lista_b"
    df_reviews['Detailing_pred'] = np.where(df_reviews['Detailing_pred'].isin(df_classes['Detalhamento']), df_reviews['Detailing_pred'], np.nan)

    return df_reviews












