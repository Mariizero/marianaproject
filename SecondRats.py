import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Função para ler o arquivo .txt e separá-lo em múltiplos arquivos Excel
def txt_to_multiple_excels(txt_file, excel_base_name):
    # Lendo o arquivo .txt
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # Separando o conteúdo em duas colunas
    col1 = []
    col2 = []
    for line in lines:
        parts = line.strip().split()  # Supondo que os dados estão separados por espaços
        if len(parts) >= 2:
            col1.append(parts[0])
            col2.append(parts[1])
        else:
            col1.append(parts[0])
            col2.append('')  # Caso não tenha a segunda coluna

    # Dividindo os dados em múltiplos arquivos Excel
    max_rows = 1048576
    num_parts = len(col1) // max_rows + 1

    for i in range(num_parts):
        start = i * max_rows
        end = (i + 1) * max_rows
        df = pd.DataFrame({'Coluna1': col1[start:end], 'Coluna2': col2[start:end]})

        # Nome do arquivo Excel de saída
        excel_file = f"{excel_base_name}_part{i + 1}.xlsx"

        # Salvando o DataFrame em um arquivo Excel
        df.to_excel(excel_file, index=False)



# Exemplo de uso
txt_file = '/Users/mv940/Desktop/wetransfer_wistar-rats-data_2024-07-01_1453/WI50basalEXEMPLO.txt'  # Substitua pelo caminho do seu arquivo .txt
excel_base_name = '/Users/mv940/Desktop/'  # Nome base para os arquivos Excel de saída
txt_to_multiple_excels(txt_file, excel_base_name)

