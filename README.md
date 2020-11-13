# Apredizado-de-maquina-aplicada-a-patologias-renais
Esse repositório contém os códigos do projeto de iniciação científica com título Estudo e implementação de algoritmos 
de aprendizagem de máquina supervisionado aplicados no Diagnóstico por imagens de Patologias Renais realizado pelo
discente Pedro Antonio Fernandes da Silva e orientado por Vinicius Ponte Machado.

## Instruções
O desenvolvimento deste trabalho se baseia na metodologia descrita por Fayyad et al.
(1996). Os arquivos relacionados a cada etapa estão descritos a seguir:

1. Seleção: A base de dados original obitida foi base-100-limpa.csv e a pasta Extração de características da imagem;

2. Pré-processamento: Data cleaning.ipynb;

3. Transformação: Data transformation.ipynb;

4. Mineração de dados: pastas RandomForest e Rede neurais;

5. Interpretação/avaliação: pasta Avaliação.

## Observações
1. Pela grande quantidade de arquivos não foi posspivel incluir no repositório as imagens utilizadas
2. A modelo escolhido foi a floresta aleatória, porém foram mantidos no repositório os testes realizados de redes neurais.
Assim como também foi mantido os diversos testes de distribuições dos diagnósticos, os escolhidos são Group_by_definitive - 
GLCM e Group_by_definitive.

## Referências
FAYYAD, U. M.; PIATETSKY-SHAPIRO, G.; SMYTH, P. (1996). From Data Mining to Knowledge in Databases. AI Magazine, [Menlo Park], v.17, n.3, mar. 1996. Disponível em: <https://www.aaai.org/ojs/index.php/aimagazine/article/view/1230> . Acesso em: 19 fev. 2020.
