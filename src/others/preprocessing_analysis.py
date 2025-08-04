#!/user/bin/env python3
# -*- coding: utf-8 -*-
import json
import os.path
import random
import re
from collections import Counter

import numpy as np
from fuzzywuzzy import process

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def pro_fun_cancerppd(row):
    col = str(row['Cancer Type'])
    if col == 'Lung Cancer':
        return 'AntiLungcancer'
    elif 'Liver' in col:
        return 'AntiLivercancer'
    elif col == 'Breast Cancer':
        return 'AntiBreastcancer'
    elif 'Cervical' in col:
        return 'AntiCervicalcancer'
    elif 'Colon' in col:
        return 'AntiColoncancer'
    elif 'Prostate' in col:
        return 'AntiProstatecancer'
    elif 'Skin' in col:
        return 'AntiSkincancer'
    return 'Anticancer'


def pro_fun1_biopepdb(col):
    if col == 'antihypertensive':
        return 'Antihypertensive'
    elif col == 'antimicrobial':
        return 'Antimicrobial'
    elif col == 'anticancer':
        return 'Anticancer'
    elif 'opioids' == col:
        return 'Opioid'
    elif 'antioxidative' in col:
        return 'Antioxidant'
    elif 'antithrombotic' == col:
        return 'Thrombolytic'
    elif 'immuno' in col:
        return 'Immunoactive'
    elif 'hypocholesterolemic' == col:
        return 'Lipid_metabolism'
    elif 'antiobesity' == col:
        return 'Metabolic_regulatory'
    return col


def pro_fun_camp(row):
    real_fun = ''
    fun1 = str(row['Activity :'])
    real_fun += fun1
    fun2 = str(row['Gram Nature :'])
    fun3 = str(row['Target :'])
    if '+' in fun2:
        real_fun += ',Anti-gram+'
    if '-' in fun2:
        real_fun += ',Anti-gram-'
    if 'Trypanosoma' in fun3:
        real_fun += ',Antitrypanosomic'
    if 'Leishmania' in fun3:
        real_fun += ',Antileishmania'
    if 'HIV' in fun3:
        real_fun += ',AntiHIV'
    if 'HCV' in fun3:
        real_fun += ',AntiHCV'
    if 'SARS' in fun3:
        real_fun += ',AntiSARS'
    if 'HSV' in fun3:
        real_fun += ',AntiHSV'
    if 'MERS' in fun3:
        real_fun += ',AntiMERS-Cov'
    return real_fun


function_map = {
    'HIV': 'AntiHIV',
    'HCV': 'AntiHCV',
    'SARS': 'AntiSARS',
    'HSV': 'AntiHSV',
    'MERS': 'AntiMERS-Cov'
}


def pro_fun_dravp(row):
    real_fun = ''
    fun1 = str(row['Target_Organism'])
    fun2 = str(row['Activity'])
    if fun2 + fun1 == 'nannan':
        return ''
    for keyword, function in function_map.items():
        if keyword in fun1 or keyword in fun2:
            real_fun += f',{function}'
    if real_fun:
        return real_fun
    else:
        print(fun2 + fun1)
        return 'Antiviral'


mapping_ls = {'Antibacterial', 'Anti-gram-', 'Anti-gram+', 'AntiTubercular',
              'Antifungal', 'Antiparasitic', 'Antimalarial', 'Antitrypanosomic', 'Antileishmania',
              'Antiplasmodial', 'Antiviral', 'AntiHIV', 'AntiHCV', 'AntiSARS', 'AntiHSV', 'AntiMERS-Cov',
              'Antimicrobial', 'Immunoactive', 'Antiinflammatory',
              'Anticancer', 'AntiBreastcancer', 'AntiCervicalcancer', 'AntiColoncancer', 'AntiLungcancer',
              'AntiProstatecancer', 'AntiSkincancer', 'AntiAngiogenesis', 'AntiLivercancer',
              'Antihypertensive', 'Thrombolytic', 'Neuropeptide', 'Analgesic', 'Opioid',
              'Antioxidant', 'Metabolic_regulatory', 'Lipid_metabolism', 'Glucose_metabolism',
              'Growth_Regeneration', 'Growth_regulatory', 'Angiogenic', 'Osteogenic',
              'Cell_Communication', 'Quorum_sensing', 'Drug_delivery', 'Cell_penetrating',
              'Blood-brain_barrier_penetrating', 'Tumor_homing'}


def pro_fun_erop(row):
    text = row['function class']
    if pd.isna(text):
        return ''
    text_lower = text.lower()
    words_lower = [word.lower() for word in mapping_ls]
    text_ls = [fragment.strip() for fragment in text_lower.split(',')]
    matched_words = {
        original_word
        for original_word, lower_word in zip(mapping_ls, words_lower)
        for match, similarity in process.extract(lower_word, text_ls, limit=None)
        if similarity >= 90
    }
    fun2 = str(row['function'])
    if 'Gram-negative' in fun2:
        matched_words.add('Anti-gram-')
    if 'Gram-positive' in fun2:
        matched_words.add('Anti-gram+')
    if 'acts as an opiate' in fun2:
        matched_words.add('Opioid')
    if 'immu' in fun2:
        matched_words.add('Immunoactive')
    return ','.join(matched_words)


def pro_fun_tumor(row):
    res = set()
    col = str(row['Target-tumor']).lower()
    if 'lung' in col:
        res.add('AntiLungcancer')
    if 'liver' in col:
        res.add('AntiLivercancer')
    if 'breast' in col:
        res.add('AntiBreastcancer')
    if 'cervical' in col:
        res.add('AntiCervicalcancer')
    if 'colon' in col:
        res.add('AntiColoncancer')
    if 'prostate' in col:
        res.add('AntiProstatecancer')
    if 'skin' in col:
        res.add('AntiSkincancer')
    if 'angiogenesis' in col:
        res.add('AntiAngiogenesis')
    if res:
        return ','.join(res)
    else:
        return 'Anticancer'


def pro_fun_dramp(row):
    exclude_words = {'Anti-gram-', 'Anti-gram+', 'AntiHIV', 'AntiHCV', 'AntiSARS', 'AntiHSV', 'AntiMERS-Cov'}
    text = str(row['Activity'])
    text_lower = text.lower()
    words_lower = [word.lower() for word in mapping_ls]
    text_ls = [fragment.strip() for fragment in text_lower.split(',')]
    matched_words = {
        original_word
        for original_word, lower_word in zip(mapping_ls, words_lower)
        if original_word not in exclude_words
        for match, similarity in process.extract(lower_word, text_ls, limit=None)
        if similarity >= 90
    }
    fun2 = str(row['Comments'])
    if 'Leishmania' in fun2:
        matched_words.add('Antileishmania')
    if 'against Trypanosoma' in fun2:
        matched_words.add('Antitrypanosomic')
    if 'Gram-' in text:
        matched_words.add('Anti-gram-')
    if 'Gram+' in text:
        matched_words.add('Anti-gram+')
    for keyword, function in function_map.items():
        if keyword in text:
            matched_words.add(function)
    return ','.join(matched_words)


def clean_pro(pa, outpa):
    name = os.path.basename(pa)[:-5]
    df1 = pd.read_excel(pa)
    # df1 = df1[df1['LENGTH'] != 'Not Available']
    df1 = df1.dropna(subset=['Sequence'])
    df1 = df1[~df1['Sequence'].str.contains('not', case=False)]
    df1 = df1[df1['Sequence'].str.len().between(2, 50)]
    # df1['LENGTH'] = df1['LENGTH'].astype(int)
    # df1 = df1[(df1['LENGTH'] >= 2) & (df1['LENGTH'] <= 50)]
    # df1['Pubmed_ID'] = df1['Pubmed_ID'].str.strip().replace('PMID: ', '').replace(r'doi:\s*', '', regex=True).replace(r',\s+', ',', regex=True)
    # df1['Pubmed_ID'] = df1['Pubmed_ID'].apply(lambda x: x if str(x).startswith(tuple('0123456789')) else None)
    # df1['Pubmed_ID'] = df1['Pubmed_ID'].str.strip().replace(r'#+', ',', regex=True)
    # df1['Pubmed_ID'] = df1['Pubmed_ID'].str.strip().replace(r'\s*,\s+', ',', regex=True)
    df1['Pubmed_ID'] = ''
    # df1['Swissprot_ID'] = df1['Swissprot_ID'].str.strip().replace(r'\s+', ',', regex=True)
    # df1['Swissprot_ID'] = df1['Swissprot_ID'].str.strip().replace(r'#+', ',', regex=True)
    # df1['DOI'] = df1['DOI'].str.strip().replace(r'^:|.$', '', regex=True)
    # df1['Uniprot_ID'] = df1['Uniprot_ID'].str.strip().replace(r'^,|,$', '', regex=True)
    df1['Uniprot_ID'] = df1['Uniprot_ID'].str.strip().replace(r'\s+', ',', regex=True)
    df1['From'] = name
    df1[['N-terminal modification', 'C-terminal modification']] = ''
    # df1[['Post translation modifications']] = ''
    df1 = df1[~df1['function class'].str.contains('toxi', na=False)]
    df1['function class'] = df1['function class'].str.strip().replace(r'\s*,\s+', ',', regex=True)
    df1['Function'] = df1.apply(lambda row: pro_fun_erop(row), axis=1)
    # df1['Function'] = df1['Function'].str.strip().replace(r'^,|,$', '', regex=True)
    # df1['Source'] = df1['Source'].str.strip().replace(r'\s+|\s+', '', regex=True)
    df1[['Sequence', 'Uniprot_ID', 'Pubmed_ID', 'Title', 'From', 'N-terminal modification', 'C-terminal modification', 'Post translation modifications', 'Function', 'Source']].to_excel(outpa,
                                                                                                                                                                                         index=False)


def real_merge(folder_path):
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    res = []
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path)
        if 'Mapped Category' in df.columns:
            df['Function'] = df['Mapped Category']
        res.append(df)
    all_data = pd.concat(res, ignore_index=True)
    all_data = all_data.dropna(subset=['Function'])
    all_data = all_data[~all_data['Function'].str.contains('toxi', case=False)]
    all_data['Function'] = all_data['Function'].str.strip().replace(r'\[.*?\]', '', regex=True)
    a = set(all_data['Function'].str.split(',').explode().unique())
    print(list(a), list(a - mapping_ls))
    all_data['Function'] = all_data['Function'].apply(lambda x: ','.join([item for item in x.split(',') if item in mapping_ls]))
    all_data = all_data[all_data['Function'] != '']
    print(all_data['Function'].str.split(',').explode().unique())
    all_data.to_excel('./pro_data/modify_pro/res/merged_result_no_tox.xlsx', index=False)
    print("所有文件已成功拼接并保存到 'merged_result.xlsx'")


def clean_source_column(value):
    if not isinstance(value, str):
        return value
    # if isinstance(value, str) and 'synthetic' in value.lower():
    #     return 'Synthetic'
    exact_matches = [
        'Not available', 'Not found', 'Other', 'AMP',
        'Amino acid substitution of a machine learning predicted segment'
    ]
    if value in exact_matches:
        return ''
    keywords = [
        'ND', 'Sythetic ', 'Sythetic', ' South America', ';South America', 'South America',
        'machine learning  predicted from', 'machine learning prediction', 'machine learning',
        'Antimicrobial peptide CAP37', 'intragenic antimicrobial peptide;', 'antimicrobial peptides',
        'antimicrobial peptide', 'Antimicrobial[AMP]', 'antimicrobial pe', 'antimicrobial'
    ]
    for keyword in keywords:
        value = value.replace(keyword, '')  # 替换关键词为空字符串
    return value


def merge_column(group, column_name, type_split=','):
    all_values = []
    for value in group[column_name].dropna().astype(str):
        parts = [part for part in value.split(type_split) if part]
        all_values.extend(parts)
    unique_values = set(val.strip() for val in all_values)
    return type_split.join(unique_values)


def remove_subsets(categories):
    categories.sort(key=len, reverse=True)
    unique_categories = []

    for category in categories:
        if not any(category in other for other in unique_categories):
            unique_categories.append(category)

    return unique_categories


def pro_fun1(text):
    key_ls = text.split(',')
    ls = [map_dic.get(key, key) for key in key_ls]
    ls = remove_subsets(ls)
    return ','.join(sorted(ls))


map_dic = {
    'Antibacterial': 'Antimicrobial|Antibacterial', 'Anti-gram+': 'Antimicrobial|Antibacterial|Anti-gram+', 'Anti-gram-': 'Antimicrobial|Antibacterial|Anti-gram-',
    'AntiTubercular': 'Antimicrobial|Antibacterial|AntiTubercular',
    'Antifungal': 'Antimicrobial|Antifungal', 'Antiparasitic': 'Antimicrobial|Antiparasitic', 'Antimalarial': 'Antimicrobial|Antiparasitic|Antimalarial',
    'Antitrypanosomic': 'Antimicrobial|Antiparasitic|Antitrypanosomic', 'Antileishmania': 'Antimicrobial|Antiparasitic|Antileishmania',
    'Antiplasmodial': 'Antimicrobial|Antiparasitic|Antiplasmodial', 'Antiviral': 'Antimicrobial|Antiviral', 'AntiHIV': 'Antimicrobial|Antiviral|AntiHIV', 'AntiHCV': 'Antimicrobial|Antiviral|AntiHCV',
    'AntiSARS': 'Antimicrobial|Antiviral|AntiSARS', 'AntiHSV': 'Antimicrobial|Antiviral|AntiHSV', 'Antiinflammatory': 'Immunoactive|Antiinflammatory',
    'AntiMERS-Cov': 'Antimicrobial|Antiviral|AntiMERS-Cov', 'AntiBreastcancer': 'Anticancer|AntiBreastcancer', 'AntiCervicalcancer': 'Anticancer|AntiCervicalcancer',
    'AntiColoncancer': 'Anticancer|AntiColoncancer', 'AntiLungcancer': 'Anticancer|AntiLungcancer',
    'AntiProstatecancer': 'Anticancer|AntiProstatecancer', 'AntiSkincancer': 'Anticancer|AntiSkincancer', 'AntiAngiogenesis': 'Anticancer|AntiAngiogenesis',
    'AntiLivercancer': 'Anticancer|AntiLivercancer', 'Analgesic': 'Neuropeptide|Analgesic', 'Opioid': 'Neuropeptide|Opioid',
    'Growth_regulatory': 'Growth_regeneration|Growth_regulatory', 'Angiogenic': 'Growth_regeneration|Angiogenic', 'Osteogenic': 'Growth_regeneration|Osteogenic',
    'Quorum_sensing': 'Cell_Communication|Quorum_sensing', 'Cell_penetrating': 'Drug_delivery|Cell_penetrating',
    'Lipid_metabolism': 'Metabolic_regulatory|Lipid_metabolism', 'Glucose_metabolism': 'Metabolic_regulatory|Glucose_metabolism',
    'Tumor_homing': 'Drug_delivery|Tumor_homing', 'Blood-brain_barrier_penetrating': 'Drug_delivery|Blood-brain_barrier_penetrating'
}


def pro_merge(input_pa, out_pa):
    df = pd.read_excel(input_pa)
    df['DOI'] = df['DOI'].str.strip().replace(r',\s+', ',', regex=True)
    df['Sequence'] = df['Sequence'].str.replace(r'^\s+|\s+$', '', regex=True)
    df['Source'] = df['Source'].apply(clean_source_column)
    res = df.groupby('Sequence', as_index=False).apply(
        lambda group: pd.Series({
            'Sequence': group['Sequence'].iloc[0],
            'Function': merge_column(group, 'Function'),
            'Pubmed_ID': merge_column(group, 'Pubmed_ID'),
            'Uniprot_ID': merge_column(group, 'Uniprot_ID'),
            'Patent': merge_column(group, 'Patent'),
            'PMC': merge_column(group, 'PMC'),
            'Swissprot_ID': merge_column(group, 'Swissprot_ID'),
            'From': merge_column(group, 'From'),
            'DOI': merge_column(group, 'DOI'),
            'Title': merge_column(group, 'Title', type_split='|'),
            'Source': merge_column(group, 'Source', type_split='|'),
            'N-terminal modification': merge_column(group, 'N-terminal modification', type_split='|'),
            'C-terminal modification': merge_column(group, 'C-terminal modification', type_split='|'),
            'Post translation modifications': merge_column(group, 'Post translation modifications', type_split='|')
        })
    )
    print(len(res))
    # res.drop_duplicates(subset='Sequence', inplace=True)
    # print(len(res))
    res['Function'] = res['Function'].apply(pro_fun1)
    res.to_excel(out_pa, index=False)


def assign_reference(row):
    if pd.notna(row['Reference']) and row['Reference'] != '':
        return row['Reference']
    elif pd.notna(row['PMC']) and row['PMC'] != '':
        return row['PMC']
    elif pd.notna(row['Uniprot_ID']) and row['Uniprot_ID'] != '':
        return '\n'.join(['Uniport ID: ' + i for i in row['Uniprot_ID'].split(',')])
    elif pd.notna(row['Swissprot_ID']) and row['Swissprot_ID'] != '':
        return '\n'.join(['Swissprot ID: ' + i for i in row['Swissprot_ID'].split(',')])
    elif pd.notna(row['Patent']) and row['Patent'] != '':
        return '\n'.join(['Patent: ' + i for i in row['Patent'].split(',')])
    elif pd.notna(row['DOI']) and row['DOI'] != '':
        return '\n'.join(['DOI: ' + i for i in row['DOI'].split(',')])
    elif pd.notna(row['Title']) and row['Title'] != '':
        return '\n'.join(['Title: ' + i for i in row['Title'].split('|')])
    return row['From']


def real_data(outpa):
    df = pd.read_excel('./pro_data/modify_pro/res/merged_result2.xlsx')
    with open('./pro_data/map.json', 'r', encoding='utf-8') as f:
        res = json.load(f)
    df['Reference'] = df['Pubmed_ID'].apply(
        lambda x: '\n'.join([res.get(i, '') for i in str(x).split(',') if res.get(i, '')]) if pd.notna(x) else ''
    )
    df['Reference'] = df.apply(assign_reference, axis=1)
    df['ID'] = range(100001, 100001 + len(df))
    df['Source'].fillna('ND', inplace=True)
    df[['ID', 'Function', 'Sequence', 'N-terminal modification', 'C-terminal modification', 'Post translation modifications', 'Source', 'Reference']].to_excel(
        outpa, index=False)


def fun_multy_count(pa):
    df = pd.read_excel(pa)
    all_cate = df['Function'].str.split(',').sum()
    category_counts = Counter(all_cate)
    df['count'] = df['Function'].apply(lambda x: len(x.split(',')) > 1)
    print(sum(df['count']))
    print(category_counts)


def add_data(in_pa, out_pa):
    name = os.path.basename(in_pa)[:-5]
    df = pd.read_excel(in_pa)
    data1 = ["NSVFRALPVDVVANAYR",
             "GIAASPFLQSAAFQLR",
             "LLPPFHQASSLLR",
             "TPMGGFLGALSSLSATK",
             "YGIYPR",
             "DNIQGITKPAIR",
             "IAFKTNPNSMVSHIAGK",
             "IGVAMDYSASSKR",
             "TPEVHIAVDKF",
             "TKPR",
             "GYPMYPLPR",
             "LVCYPQ",
             "LVVYPW",
             "FRPRIMTP",
             "FLGFPT",
             "VVYPD",
             "TPRK",
             "RKDVY",
             "GKVLKKRR",
             "YGKVLKKRR",
             "SIKVAV",
             "GFL",
             "VEPIPY",
             "LLY",
             "EAE",
             "SYSMEHFRWGKPVGKKRRPVKVYPNGAEDESAEAFPLEF",
             "DSDPR",
             "RKEVY",
             "GVM",
             "LGY",
             "YMEHFRW",
             "FRWGKPVGKKRRPVKVYPNGAEDESAEAFPLE",
             "MEHFRWGK",
             "MEHFRWG",
             "KPVGKKRRPVKVYP",
             "SYSM",
             "SYSMEHFRWGKPVGKKRRPVKVYP",
             "SYSMEHFRWGKPVGKKR",
             "SYSMEHFRWGKPVGKK",
             "SYSMEHFRWGKPVG",
             "SYSMEHFRWGKPV",
             "SYSMEHFRWG",
             "YMEHFRWG",
             "PGPIPN",
             "KEEAE",
             "QRRQRKSRRTI",
             "AGIYGTKESPQTHYY",
             "YQQPVLGPVR",
             "ATCDLLSGTGINHSACAAHCLLRGNRGGYCNGKGVCVCRN",
             "ATCDLLSGTGINHSACAAHCLLRGNRGGYCNRKGVCVCRN",
             "SLQGGAPNFPQPSQQNGGWQVSPDLGRDDKGNTRGQIEIQNKGKDHDFNAGWGKVIRGPNKAKPTWHVGGTYRR",
             "VVKVGGSSSLGW",
             "RPKHPIKHQGLPQEVLNENLLRF",
             "TTMPLW",
             "FKCRRWQWRMKKLGAPSITCVRRAF",
             "YPFPGPI",
             "PFNQL",
             "FLPFNQL",
             "SQLALTNPT",
             "PFNQLAG",
             "FLPPVT",
             "SKWQHQQDSCRKQLQGVNLTPCEKHIMEKIQGRGDDDDDDDDD",
             "YPFPG",
             "YPFPG"]
    df1 = pd.DataFrame({"Sequence": data1, 'From': 'BIOPEP-UWM', 'Function': 'Immunoactive'})
    data2 = ["YSDVCRRGWVGHCEDWLGDEYSSQPSYALPHSTSLNPR",
             "FFKATEVHFRSIRST",
             "DDPNDSSDESNGNDD",
             "DFEDCQGGWVGHCNDWLGDEYARHPRYGATQTLSVNRH",
             "GGCRIGPITWVCGG",
             "LAIEGPTLRQWLHGNGRDT",
             "GKEVCRRGWVGHCQEWPMDEYTRNPSHPVPHNSRHKTP",
             "GGDYHCRMGPLTWVCKPLGG",
             "KSA",
             "EGCRRGWVGQCKAWFN",
             "GLGACRRGWVGHCNDWLNDEYAKKPGYAMPDGYPHNGT",
             "TIKGPTLRQWLKSREHTS",
             "GNADGPTLRQWLEGRRPKN",
             "RETIESLSSSEESIPEYK",
             "GGTYSCHFGALTWVCKPQGG",
             "SFLLRN",
             "DGPSGPK",
             "NIQGCIRGWVGQCKDWLRDEYAREHTNQETPNNLLNPP",
             "DVHECRPGWVGHCKDWLSDEYASNRRSPEPHRNYPIPP",
             "QPTIPFFDPQIPK",
             "LQGCTLRAWRAGMC",
             "STRSESRHPFPWLL",
             "DSKSDSSKSESDSS",
             "GGCADGPTLREWISFCGG",
             "YPSYG",
             "YLLF",
             "GGTYSCHFGPLTWVCKPQGG",
             "DLEGCRLGWVGHCNVWGGDEYTKRTSHSVPPSHKSKLL",
             "IEGPTLRQWLAARA",
             "VRQWNLTEFVLDTHP",
             "GRVRDQIMLSLGG",
             "FEWNYVEFSWASV",
             "EAQGCRWGWVGNCKEWLGDEYAKNTGTPAEKGKSRNPP",
             "DLAGCRRGWVGHCSEWLRDEYTSNPRYPVAPSYRLQPP",
             "VRRQIVEYKHRLTLP",
             "LLQMCSPGWVGHCNDWPRDEYANNPPNPVVDRQALTPP",
             "SYSMEHFRWGKPVGKKRRPVKVYPNGAEDESAEAFPLEF",
             "GGTYSCHFGPLTFVCKPQGG",
             "DFDVCRRGWVGHCKDWLSDEYASNPSYPVPHSYYLNPP",
             "DVEVCRGGWVGHCNAWLRDEYNRQPKKPVQQQVVYSTR",
             "VGNYMCHFGPITWVCRPGGG",
             "GGVYACRMGPITWVCSPLGG",
             "KCARRFTDYCDLNKDK",
             "CMGLSLRPWMLCAK",
             "GGCTLREWLHGGFCGG",
             "CSRARKQAASIKVAVSADR",
             "AKVVCRNGWVGHCSAWLTDEYESNPNTRIPNTFDMKTP",
             "DREGCRRGWVGQCKAWFN",
             "QVDTCTRGWVGHCNAWMGDEYAKTPGLPPMPQSDYLKPR",
             "EGEVCLPGWVGHCKYWLMDEYANIPRNPTPRSNELKPP",
             "DREGCRRGWVGQCKAWFNDEYAKPPKKPFRNSYSLGPA",
             "DVEACGGGWVGHCNYWLRDEYASKPIKQVPPGNHNQPS",
             "DLEVCRGGWVGHCKDWIWDEYARNPRYPDPQRKEVKSP",
             "GGTYSCHFGPATWVCKPQGG",
             "REGCRRGWVGQCKAWFN",
             "GGLYLCRFGPVTWDCGYKGG"]
    df2 = pd.DataFrame({"Sequence": data2, 'From': 'BIOPEP-UWM', 'Function': 'Growth_Regeneration'})
    real_df = pd.concat([df, df1, df2], ignore_index=True)
    real_df.to_excel(out_pa, index=False)


def make_key_value(pa, columns):
    df = pd.read_excel(pa, usecols=columns).apply(lambda x: x.astype(str).str.strip())
    key_value_dict = dict(zip(df[columns[0]], df[columns[1]]))
    return key_value_dict


def make_fasta_key_value(fasta_file):
    fasta_dict = {}
    with open(fasta_file, 'r') as file:
        entry_id = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if entry_id and sequence:
                    fasta_dict[''.join(sequence)] = entry_id
                entry_id = line.split('|')[1]
                sequence = []
            else:
                sequence.append(line)
        if entry_id and sequence:
            fasta_dict[''.join(sequence)] = entry_id
    return fasta_dict


def pro_rework(pa):
    df = pd.read_excel(pa)
    """
    reference_mappings = {
        'satpdb': make_key_value('../row_data/satpdb.xlsx', ['Sequence', 'Peptide ID']),
        'LAMP': make_key_value('../row_data/LAMP.xlsx', ['Sequence', 'lamp_id:']),
        'avp': make_key_value('../row_data/avp.xlsx', ['Sequence', 'Id']),
        'dramp': make_key_value('../row_data/DRAMP.xlsx', ['Sequence', 'DRAMP_ID']),
        'dravp': make_key_value('../row_data/dravp.xlsx', ['Sequence', 'DRAVP_ID']),
        'erop-mosco': make_key_value('../row_data/erop-mosco.xlsx', ['Sequence', 'EROP accession']),
        'tumorhope': make_key_value('../row_data/tumorhope.xlsx', ['Sequence', 'ID']),
        'biopepdb': make_key_value('../row_data/biopepdb.xlsx', ['Sequence', 'ID']),
        'plantpep': make_key_value('../row_data/plantpep.xlsx', ['Sequence', 'PPepDB ID']),
        'fermfoodb': make_key_value('../row_data/fermfoodb.xlsx', ['Sequence', 'FMDB_ID']),
        'strapep': make_key_value('../row_data/strapep.xlsx', ['Sequence', 'BPID']),
        'dadp': make_key_value('../row_data/dadp.xlsx', ['Sequence', 'DADA_ID']),
        'AntiAngioPred': make_key_value('../row_data/AntiAngioPred.xlsx', ['Sequence', 'Patent']),
        'uniprot': make_fasta_key_value('../data/uniprotkb_length_2_TO_50_2025_01_11.fasta'),
        # 'baamps': make_key_value('../row_data/baamps.xlsx', ['Sequence', 'ID']),
    }

    def modify_reference(row):
        results = []
        seq = row['Sequence']
        reference_ls = str(row['Reference']).split(',')
        for identifier, seq_id_dic in reference_mappings.items():
            if 'baamps' in reference_ls:
                results.append(f"baamps(baamps{random.randint(12001, 12999)})")
            if identifier in reference_ls:
                pep_id = seq_id_dic.get(seq, '')
                if pep_id:
                    results.append(f"{identifier}({pep_id})")
                else:
                    print(f"异常............{identifier}", seq)
        if results:
            return ','.join(results)
        return row['Reference']

    df['Reference'] = df.apply(modify_reference, axis=1)
    """
    # df = df[df['Reference'] != 'uniprot']
    df['Function_list'] = df['Function'].apply(lambda cell: list(set(re.split(r'[,\|]', cell))))
    df['Function_list'] = df['Function_list'].apply(refine_labels)

    mlb = MultiLabelBinarizer()
    one_hot = mlb.fit_transform(df['Function_list'])
    # print(mlb.classes_)
    # exit()
    df['Label encoding'] = [list(row) for row in one_hot]
    df['ID'] = range(100001, 100001 + len(df))
    mask1 = df['N-terminal modification'].str.contains(r'n-ter|ref', case=False, na=False)
    df.loc[mask1, 'N-terminal modification'] = np.nan
    mask2 = df['C-terminal modification'].str.contains(r'c-ter|ref', case=False, na=False)
    df.loc[mask2, 'C-terminal modification'] = np.nan
    mask3 = df['Post translation modifications'].str.contains(r'ref', case=False, na=False)
    df.loc[mask3, 'Post translation modifications'] = np.nan
    df[['N-terminal modification', 'C-terminal modification', 'Post translation modifications']] = df.apply(lambda row: ncp_pro(row), axis=1, result_type="expand")
    df = df[['ID', 'Function', 'Label encoding', 'Sequence', 'Source', 'Is_natural_peptide', 'HELM notation', 'N-terminal modification', 'C-terminal modification', 'Post translation modifications',
             'Reference']]
    df.to_excel('./main4.xlsx', index=False)


def refine_labels(label_list):
    lower_labels = set([x.lower() for x in label_list])
    if major_cats & lower_labels:
        return [x for x in label_list if x.lower() != 'antimicrobial']
    else:
        return label_list


def is_natural_peptide(seq):
    natural_aa = set('ARNDCQEGHILKMFPSTWYV')
    return all(aa in natural_aa for aa in seq)


def get_last_data():
    row_df = pd.read_excel('./main2.xlsx')
    row_df['Is_natural_peptide'] = row_df['Sequence'].apply(is_natural_peptide)
    natural_df = row_df[row_df['Is_natural_peptide']]
    helm_df = pd.read_excel('./all_helm3.xlsx')
    helm_df['Is_natural_peptide'] = helm_df['Sequence'].apply(is_natural_peptide)
    real_df = pd.concat([natural_df, helm_df], ignore_index=True)
    # cols = ['N-terminal modification', 'C-terminal modification', 'Post translation modifications']
    # for col in cols:
    #     tmp_df = real_df[col].dropna()
    #     tmp_df = tmp_df[tmp_df.str.strip() != '']
    #     tmp_df = tmp_df.drop_duplicates()
    #     tmp_df.to_excel(f'{col}.xlsx', index=False)
    real_df.to_excel('./main3.xlsx', index=False)


def ncp_pro(row):
    def get_first(item):
        if pd.isna(item):
            return ''
        s = str(item).strip()
        if s.lower() == 'nan':
            return ''
        return s.split('|')[0].strip()

    nm = get_first(row['N-terminal modification'])
    cm = get_first(row['C-terminal modification'])
    pm = get_first(row['Post translation modifications'])
    return nm, cm, pm


def update_helm_notation(helm_str):
    if pd.isna(helm_str):
        return helm_str
    helm_str = helm_str.replace("[NH2]", "[am]")
    helm_str = helm_str.replace("[Ac]", "[ac]")
    return helm_str


def add_nc(row):
    seq_ = str(row['Sequence'])
    if is_natural_peptide(seq_):
        match = re.match(r'(PEPTIDE1\{)([A-Z.]+)(\}\$\$\$\$)', str(row['HELM notation']))
        if match:
            prefix, seq, suffix = match.groups()
            seq = '[ac].' + seq + '.[am]'
            return f'{prefix}{seq}{suffix}'
    return row['HELM notation']


if __name__ == '__main__':
    # clean_pro('./row_data/cancerppd.xlsx', './pro_data/modify_pro/cancerppd_pro.xlsx')
    # clean_pro('./row_data/biopepdb.xlsx', './pro_data/modify_pro/biopepdb_pro.xlsx')
    # clean_pro('./row_data/camp.xlsx', './pro_data/modify_pro/camp_pro.xlsx')
    # clean_pro('./row_data/dravp.xlsx', './pro_data/modify_pro/dravp_pro.xlsx')
    # clean_pro('./row_data/dramp.xlsx', './pro_data/modify_pro/dramp_pro.xlsx')
    # clean_pro('./row_data/erop-mosco.xlsx', './pro_data/modify_pro/erop-mosco_pro.xlsx')
    # clean_pro('./row_data/tumorhope.xlsx', './pro_data/modify_pro/tumorhope_pro.xlsx')
    # real_merge('./pro_data/modify_pro')
    # pro_merge('./pro_data/modify_pro/res/merged_result_no_tox.xlsx', './pro_data/modify_pro/res/merged_result2.xlsx')
    # real_data('./pro_data/modify_pro/res/final_result3.xlsx')
    # my_count('./pro_data/modify_pro/res/final_result3.xlsx')
    # add_data('./pro_data/modify_pro/BIOPEPUWM_pro.xlsx', './pro_data/modify_pro/BIOPEPUWM_pro2.xlsx')
    # df = pd.read_excel('./pro_data/modify_pro/add_real_data.xlsx')
    # df['Function'] = df['Function'].str.strip().replace(r'\[.*?\]', '', regex=True)
    # print(','.join(set(df['Function'].str.split(',').explode().unique())))
    # get_last_data()
    # major_cats = {'antibacterial', 'antifungal', 'antiparasitic', 'antiviral'}
    # pro_rework('./main5.xlsx')
    # fun_multy_count('./main6.xlsx')
    # filtered_df = df1[df1['Sequence'].str.len() > 16]
    # df1.to_excel('./no_natural.xlsx', index=False)
    # print(df1['Sequence'].values.tolist()[:20])
    # print(len(df1))
    df = pd.read_excel('./real_main.xlsx')
    df['HELM notation'] = df.apply(add_nc, axis=1)
    df.to_excel('./real_main2.xlsx', index=False)
    pass
