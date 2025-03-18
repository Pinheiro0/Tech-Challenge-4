🎥 Análise de Vídeo com IA
Este projeto implementa um sistema avançado de análise de vídeo que combina três funcionalidades principais usando Inteligência Artificial:

✅ Reconhecimento Facial
✅ Detecção de Emoções
✅ Reconhecimento de Atividades

🚀 Funcionalidades
🔍 Reconhecimento Facial
Detecta faces em cada frame do vídeo.
Identifica pessoas conhecidas comparando com imagens de referência.
Marca as faces detectadas com retângulos verdes.
Exibe o nome da pessoa identificada ou "Não Identificado".
😊 Detecção de Emoções
Analisa a emoção predominante em cada face detectada.
Suporta as seguintes emoções:
Feliz 😃
Triste 😢
Irritado 😠
Medo 😨
Surpreso 😮
Neutro 😐
Nojo 🤢
⚠️ Detecção de Anomalias
Identifica comportamentos anômalos no vídeo.
Considera anomalia quando:
A confiança na detecção de atividade é menor que 30%.
Marca frames anômalos com alerta visual.
Gera estatísticas de anomalias no relatório final.
🏃 Reconhecimento de Atividades
Identifica a atividade sendo realizada no vídeo.
Suporta diversas atividades como:
Dançando 💃
Dormindo 💤
Aplaudindo 👏
Bebendo 🥤
Comendo 🍽
E muitas outras...
Exibe a atividade detectada com nível de confiança.
📦 Requisitos
🛠 Dependências
Python 3.x
OpenCV
DeepFace
Transformers
PyTorch
Torch
tqdm
numpy
tensorflow
🔍 Modelos Pré-treinados
O sistema baixa automaticamente os seguintes modelos:

deploy.prototxt
res10_300x300_ssd_iter_140000.caffemodel
openface.nn4.small2.v1.t7
📂 Estrutura do Projeto
Copiar
Editar
projeto_analise_video/
│── face_emotion_activities.py
│── images/
│── output_combined.mp4
│── README.md
▶️ Como Usar
🎬 Preparação
1️⃣ Coloque o vídeo a ser analisado como video.mp4 na pasta do projeto.
2️⃣ Adicione fotos das pessoas a serem reconhecidas na pasta images.
3️⃣ Nomeie as fotos com o nome da pessoa (ex: "João.jpg").

🚀 Execução
Execute o seguinte comando no terminal:

bash
Copiar
Editar
python face_emotion_activities.py
🎯 Saída
O sistema gerará:
✅ output_combined.mp4 (vídeo processado)
✅ output_combined_resumo.md (relatório detalhado)

O vídeo de saída mostrará:

📌 Faces detectadas com nomes
📌 Emoções identificadas
📌 Atividade atual com nível de confiança
📌 Alertas de anomalias quando detectadas
O relatório incluirá:

📊 Estatísticas gerais do vídeo
📊 Contagem de anomalias detectadas
📊 Distribuição de atividades e emoções
📊 Lista de pessoas reconhecidas
⚙️ Configurações
O sistema possui alguns parâmetros configuráveis:

Parâmetro	Descrição	Padrão
confidence_threshold	Limiar para detecção facial	0.7
face_size	Tamanho padrão para processamento facial	96
buffer_size	Frames para suavização temporal	5
anomaly_threshold	Limiar para detecção de anomalias	0.3
⚠️ Limitações
🔸 Requer boa iluminação para melhor detecção facial.
🔸 O desempenho pode variar dependendo da qualidade do vídeo.
🔸 Necessita de recursos computacionais adequados para processamento em tempo real.
🔧 Tecnologias Utilizadas
🧠 PyTorch - Modelos de deep learning
📹 OpenCV - Processamento de imagem e vídeo
😊 DeepFace - Análise de emoções
📚 Transformers - Reconhecimento de atividades
⚡ CUDA - Suporte opcional para aceleração por GPU
🤝 Contribuição
Sinta-se à vontade para contribuir com o projeto através de:
✅ Relatórios de bugs 🐞
✅ Sugestões de melhorias 💡
✅ Pull requests 🔄

📝 Licença
Este projeto está sob a licença MIT.
