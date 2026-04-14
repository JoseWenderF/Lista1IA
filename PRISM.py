import pandas as pd
import numpy as np

# --- IMPLEMENTAÇÃO DO ALGORITMO PRISM ---

def discretize_df(df, target_col):
    
    # Transforma colunas numéricas em faixas: Baixo, Médio e Alto.

    df_disc = df.copy()
    for col in df_disc.columns:
        if col == target_col:
            continue
        # Se a coluna for numérica e tiver muitos valores únicos, discretizamos
        if df_disc[col].dtype != 'object' and df_disc[col].nunique() > 10:
            try:
                df_disc[col] = pd.qcut(df_disc[col], q=3, labels=['Baixo', 'Médio', 'Alto'], duplicates='drop')
            except:
                df_disc[col] = pd.cut(df_disc[col], bins=3, labels=['Baixo', 'Médio', 'Alto'])
    return df_disc

def prism_algorithm(df, target_col, max_rules_per_class=5):
    
    #Implementação do Indutor de Regras PRISM.
    
    classes = df[target_col].unique()
    all_rules = {}
    
    for cls in classes:
        rules_list = []
        subset = df.copy()
        
        # Enquanto houver instâncias da classe e não atingirmos o limite de regras
        while len(subset[subset[target_col] == cls]) > 0 and len(rules_list) < max_rules_per_class:
            current_rule = []
            temp_df = subset.copy()
            
            # Constrói a regra adicionando um termo por vez para maximizar a precisão
            while len(temp_df[temp_df[target_col] != cls]) > 0:
                best_prec = -1
                best_attr_val = None
                best_count = 0
                
                for col in temp_df.columns:
                    if col == target_col: continue
                    
                    for val in temp_df[col].unique():
                        coverage = temp_df[temp_df[col] == val]
                        if len(coverage) == 0: continue
                        
                        precision = len(coverage[coverage[target_col] == cls]) / len(coverage)
                        count = len(coverage[coverage[target_col] == cls])
                        
                        # Critério de seleção: Maior precisão, depois maior cobertura
                        if precision > best_prec:
                            best_prec = precision
                            best_attr_val = (col, val)
                            best_count = count
                        elif precision == best_prec and count > best_count:
                            best_attr_val = (col, val)
                            best_count = count
                
                if best_attr_val is None: break
                
                current_rule.append(best_attr_val)
                temp_df = temp_df[temp_df[best_attr_val[0]] == best_attr_val[1]]

            if not current_rule: break
                
            # Calcula cobertura final da regra no subset atual
            rule_str = " AND ".join([f"{attr} == {val}" for attr, val in current_rule])
            coverage_full = subset.copy()
            for attr, val in current_rule:
                coverage_full = coverage_full[coverage_full[attr] == val]
            
            rules_list.append({
                'rule': rule_str,
                'precision': best_prec,
                'coverage': len(coverage_full)
            })
            
            # Remove as instâncias cobertas pela regra (Estratégia Separar-e-Conquistar)
            subset = subset.drop(coverage_full.index)
            
        all_rules[cls] = rules_list
        
    return all_rules

def print_rules(title, rules_dict, target_name):
    print(f"\n{'='*20} {title} {'='*20}")
    for cls, rules in rules_dict.items():
        print(f"\nCLASSE ALVO: {cls}")
        for i, r in enumerate(rules):
            print(f" Regra {i+1}: IF {r['rule']} THEN {target_name} = {cls}")
            print(f"          [Precisão: {r['precision']:.2%}, Cobertura: {r['coverage']} instâncias]")


# Execução de bases de dados

try:
    # 1. Processando Diabetes
    df_diabetes = pd.read_csv('diabetes.csv')
    diabetes_disc = discretize_df(df_diabetes, 'Outcome')
    diabetes_rules = prism_algorithm(diabetes_disc, 'Outcome')
    print_rules("REGRAS: DIABETES", diabetes_rules, "Outcome")

    # 2. Processando Attrition (RH)
    df_attrition = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    # Removendo colunas que atrapalhariam 
    to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df_attrition = df_attrition.drop(columns=[c for c in to_drop if c in df_attrition.columns])
    
    attrition_disc = discretize_df(df_attrition, 'Attrition')
    attrition_rules = prism_algorithm(attrition_disc, 'Attrition')
    print_rules("REGRAS: ATTRITION (RH)", attrition_rules, "Attrition")

except FileNotFoundError as e:
    print(f"Erro: Certifique-se de que os arquivos CSV estão na mesma pasta do script. {e}")