#### MVP - CLASSIFICATION GPT
import streamlit as st
import pandas as pd
import numpy as np
import xlsxwriter
import json
import base64
from io import BytesIO
import asyncio
import aiohttp
from utils import *
import time

st.title("ClassificationGPT")
st.write('Esta aplicação tem como objetivo auxiliar nas classificações de reviews com o uso de IA')
st.write()

# Inserindo arquivo de reviews
reviewSheet = st.file_uploader("Insira um arquivo .xlsx com os reviews a serem classificados (Máx: 50 reviews)")
if reviewSheet is not None:
    df_reviews = pd.read_excel(reviewSheet)

    # Lendo reviews e verificando se há mais de 100 registros
    if df_reviews.shape[0] > 50:
        st.warning("Há mais de 50 reviews nesta base, a classificação só será feita com os 50 primeiros.")

    # Filtrando os 100 primeiros reviews
    df_reviews = df_reviews.iloc[:50]

# Inserindo arquivo de classificações
classSheet = st.file_uploader("Insira um arquivo .xlsx com as Subcategorias e Detalhamentos (Máx: 50 classes p/ Subcategoria e 100 p/ Detalhamento)")
if classSheet is not None:

    # Lendo reviews e verificando se há mais de 50 registros
    df_classes = pd.read_excel(classSheet)    
    if len(df_classes['Subcategoria'].dropna()) >50:
        st.warning("Há mais de 50 Subcategorias nesta base, serão apenas considerados as 50 primeiras.")

    if len(df_classes['Detalhamento'].dropna()) >100:
        st.warning("Há mais de 100 Detalhamentos nesta base, serão apenas considerados os 100 primeiros.")
    
    # Filtrando as 30 primeriras classificações
    df_classes = df_classes.iloc[:100]

# Visualizar dados
check_reviews = st.checkbox("Visualizar Reviews")
if reviewSheet is not None:
    if check_reviews:
        st.write(df_reviews)

check_subcategory = st.checkbox("Visualizar Subcategorias")
if classSheet is not None:
    if check_subcategory:
        st.write(df_classes[['Subcategoria']].iloc[:50])

check_detail = st.checkbox("Visualizar Detalhamentos")
if classSheet is not None:
    if check_detail:
        st.write(df_classes[['Detalhamento']].iloc[:70])

############# Tratamento e preparação de dados #############
if reviewSheet and classSheet is not None:

    # Substituindo variações de nomes de reviews
    list_string = ['Text','text','TEXT','Reviews','reviews','REVIEW','REVIEWS']
    df_reviews = replace_column_with_review(df_reviews, list_string)

    # Criar lista de reviews com a string 'Comentário: ' no início
    list_reviews = make_reviews(df_reviews)

    # Particionar lotes de reviews para serem enviados em conjunto na API
    lotes_reviews = coletar_lotes(list_reviews,1)

    # Criação de contexto para o modelo. A função recebe as classes para compor o texto
    system_sentiment  = create_system_sentiment()
    system_category  = create_system_category()
    system_subcategory  = create_system_subcategory(df_classes)
    system_detail  = create_system_detail(df_classes)

############# Tratamento e preparação de dados #############
if st.button('Gerar Classificações'):

    # Request na API p/ gerar classificações
    results_detail = asyncio.run(get_chatgpt_responses(system=system_detail, lotes_reviews=lotes_reviews))
    time.sleep(18)

    results_sentiment = asyncio.run(get_chatgpt_responses(system=system_sentiment, lotes_reviews=lotes_reviews))
    time.sleep(18)

    results_subcategory = asyncio.run(get_chatgpt_responses(system=system_subcategory, lotes_reviews=lotes_reviews))
    time.sleep(18)

    results_category = asyncio.run(get_chatgpt_responses(system=system_category, lotes_reviews=lotes_reviews))
    

    

    

    # Normalização de resultados recebidos pela API
    df_results_sentiment = normalize_results(results_sentiment)
    df_results_category = normalize_results(results_category)
    df_results_subcategory = normalize_results(results_subcategory)
    df_results_detail = normalize_results(results_detail)
    df_results_sentiment.dropna(inplace=True, axis=0)
    df_results_category.dropna(inplace=True, axis=0)
    df_results_subcategory.dropna(inplace=True, axis=0)
    df_results_detail.dropna(inplace=True, axis=0)

    # Tratamento de lotes de classificação
    df_results_sentiment = clean_results(df_results_sentiment)
    df_results_category = clean_results(df_results_category)
    df_results_subcategory = clean_results(df_results_subcategory)
    df_results_detail = clean_results(df_results_detail)

    # Acrescentar classificações no df de reviews, renomear colunas, adicionar valor Genérico caso não venha classificação da API
    df_reviews_sentiment = format_results(df_reviews=df_reviews, df_results=df_results_sentiment, group="Sentiment")
    df_reviews_category = format_results(df_reviews=df_reviews, df_results=df_results_category, group="Category")
    df_reviews_subcategory = format_results(df_reviews=df_reviews, df_results=df_results_subcategory, group="Subcategory")
    df_reviews_detail = format_results(df_reviews=df_reviews, df_results=df_results_detail, group="Detailing")

    # Substituir classificações que não estão na lista por nan
    df_reviews_subcategory = replace_errors_with_nan(df_reviews=df_reviews_subcategory, df_classes=df_classes, group='Subcategory_pred', group_class='Subcategoria')
    df_reviews_detail = replace_errors_with_nan(df_reviews=df_reviews_detail, df_classes=df_classes, group='Detailing_pred', group_class='Detalhamento')

    # Concatenar dfs de grupos
    df_final = pd.concat([df_reviews_sentiment, df_reviews_category, df_reviews_subcategory, df_reviews_detail], axis=1)
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    st.write(df_final)
    st.write('Clique em Download para baixar o arquivo')
    st.markdown(get_table_download_link(df_final), unsafe_allow_html=True)


