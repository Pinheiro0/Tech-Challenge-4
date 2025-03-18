import cv2
import numpy as np
import os
import torch
from scipy.spatial.distance import cosine
from deepface import DeepFace
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
import urllib.request

class CombinedVideoAnalyzer:
    def __init__(self):
        """Initialize all models and settings."""
        self.initialize_face_models()
        self.initialize_activity_models()
        self.confidence_threshold = 0.7
        self.face_size = 96
        self.prediction_buffer = []
        self.buffer_size = 5
        self.activity_counter = {}
        self.emotion_counter = {}
        self.face_counter = {}
        self.anomaly_counter = 0
        self.anomaly_threshold = 0.3
        
    def initialize_face_models(self):
        """Load face detection and recognition models."""
        print("Loading face detection and recognition models...")
        prototxt_path = "deploy.prototxt"
        caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
        self.face_detector = cv2.dnn.readNet(prototxt_path, caffemodel_path)
        self.face_recognizer = cv2.dnn.readNetFromTorch("openface.nn4.small2.v1.t7")
        
    def initialize_activity_models(self):
        """Load activity recognition models."""
        print("Loading activity recognition models...")
        self.model_name = "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activity_model = ViTForImageClassification.from_pretrained(self.model_name).to(self.device)
        self.activity_processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.activity_model.eval()

    def detect_and_align_face(self, frame):
        """Detect and align faces in the frame."""
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            [104, 117, 123], 
            swapRB=False, 
            crop=False
        )
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        aligned_faces = []
        face_locations = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x1, y1, x2, y2 = box.astype("int")
                
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    aligned_face = cv2.resize(face_roi, (self.face_size, self.face_size))
                    aligned_faces.append(aligned_face)
                    face_locations.append((x1, y1, x2, y2))
                    
        return aligned_faces, face_locations

    def smooth_predictions(self, prediction, confidence):
        """Aplica suavização temporal às predições."""
        self.prediction_buffer.append((prediction, confidence))
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
            
        # Contar ocorrências de cada predição
        prediction_counts = {}
        total_confidence = {}
        
        for pred, conf in self.prediction_buffer:
            if pred not in prediction_counts:
                prediction_counts[pred] = 0
                total_confidence[pred] = 0
            prediction_counts[pred] += 1
            total_confidence[pred] += conf
            
        # Encontrar a predição mais frequente
        max_count = 0
        smoothed_prediction = prediction
        smoothed_confidence = confidence
        
        for pred, count in prediction_counts.items():
            if count > max_count:
                max_count = count
                smoothed_prediction = pred
                smoothed_confidence = total_confidence[pred] / count
                
        return smoothed_prediction, smoothed_confidence

    def get_face_embedding(self, face):
        """Extract facial embedding using the recognition model."""
        face_blob = cv2.dnn.blobFromImage(
            face, 
            1.0/255, 
            (self.face_size, self.face_size),
            (0, 0, 0), 
            swapRB=True, 
            crop=False
        )
        
        self.face_recognizer.setInput(face_blob)
        embedding = self.face_recognizer.forward()
        return embedding.flatten()

    def analyze_emotions(self, frame, face_location):
        """Analyze emotions in detected face using DeepFace."""
        try:
            x1, y1, x2, y2 = face_location
            face_roi = frame[y1:y2, x1:x2]
            
            analysis = DeepFace.analyze(
                face_roi, 
                actions=['emotion'],
                enforce_detection=False
            )
            
            return analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
        except Exception as e:
            print(f"Error in emotion analysis: {str(e)}")
            return None

    def process_activity(self, frame):
        """Process frame for activity recognition."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.activity_processor(images=frame_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.activity_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        confidence, prediction = torch.max(probs, 1)
        activity = self.activity_model.config.id2label[prediction.item()]
        
        # Dicionário de tradução de atividades
        activity_translations = {
            'dancing': 'dancando',
            'sleeping': 'dormindo',
            'clapping': 'aplaudindo',
            'drinking': 'bebendo',
            'laughing': 'rindo',
            'eating': 'comendo',
            'sitting': 'sentado',
            'standing': 'em pe',
            'walking': 'andando',
            'running': 'correndo',
            'jumping': 'pulando',
            'fighting': 'brigando',
            'climbing': 'escalando',
            'reading': 'lendo',
            'writing': 'escrevendo',
            'cooking': 'cozinhando',
            'talking': 'falando',
            'texting': 'mexendo no celular',
            'playing': 'brincando',
            'working': 'trabalhando',
            'exercising': 'se exercitando',
            'cleaning': 'limpando',
            'cycling': 'pedalando',
            'driving': 'dirigindo',
            'shopping': 'comprando',
            'singing': 'cantando',
            'hugging': 'abracando',
            'kissing': 'beijando',
            'pushing': 'empurrando',
            'pulling': 'puxando',
            'carrying': 'carregando',
            'throwing': 'arremessando',
            'catching': 'pegando',
            'lifting': 'levantando peso',
            'listening music': 'ouvindo musica',
            'using laptop' : 'usando laptop'
        }
        
        # Traduzir a atividade ou manter original se não houver tradução
        translated_activity = activity_translations.get(activity.lower(), activity)
        
        return translated_activity, confidence.item()

    def generate_summary(self, output_path, total_video_frames):
        """Generate a markdown summary of the video analysis."""
        total_activities = sum(self.activity_counter.values())
        total_emotions = sum(self.emotion_counter.values())
        total_faces = sum(self.face_counter.values())
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Resumo da Análise do Vídeo\n\n")
            
            # Informações gerais do vídeo
            f.write("## Informações do Vídeo\n\n")
            f.write(f"- Total de frames no vídeo: {total_video_frames}\n")
            f.write(f"- Frames analisados com sucesso: {total_activities}\n")
            if total_video_frames > 0:
                f.write(f"- Porcentagem de análise: {(total_activities/total_video_frames)*100:.1f}%\n")

            f.write(f"- Número de anomalias detectadas: {self.anomaly_counter}\n")
            
            # Nova seção para anomalias
            f.write("\n## Detecção de Anomalias\n\n")
            f.write(f"- Total de anomalias detectadas: {self.anomaly_counter}\n")
            if total_activities > 0:
                anomaly_percentage = (self.anomaly_counter / total_activities) * 100
                f.write(f"- Taxa de anomalias: {anomaly_percentage:.2f}%\n")
            f.write("- Critério: Atividade com confiança inferior a 30%\n")

            # Atividades detectadas
            f.write("\n## Atividades Detectadas\n\n")
            if self.activity_counter:
                sorted_activities = sorted(
                    self.activity_counter.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for activity, count in sorted_activities:
                    percentage = (count / total_activities) * 100
                    f.write(f"- {activity}: {percentage:.1f}% ({count} frames)\n")
            else:
                f.write("Nenhuma atividade detectada\n")
            
            # Emoções detectadas
            f.write("\n## Emoções Detectadas\n\n")
            if self.emotion_counter:
                sorted_emotions = sorted(
                    self.emotion_counter.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for emotion, count in sorted_emotions:
                    percentage = (count / total_emotions) * 100
                    f.write(f"- {emotion}: {percentage:.1f}% ({count} detecções)\n")
            else:
                f.write("Nenhuma emoção detectada\n")
            
            # Pessoas reconhecidas
            f.write("\n## Pessoas Reconhecidas\n\n")
            if self.face_counter:
                sorted_faces = sorted(
                    self.face_counter.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for name, count in sorted_faces:
                    percentage = (count / total_faces) * 100
                    f.write(f"- {name}: {percentage:.1f}% ({count} aparições)\n")
            else:
                f.write("Nenhuma pessoa reconhecida\n")
            
            # Estatísticas gerais
            f.write("\n## Estatísticas Gerais\n\n")
            f.write(f"- Total de frames no vídeo: {total_video_frames}\n")
            f.write(f"- Total de frames processados: {total_activities}\n")
            f.write(f"- Total de detecções de emoções: {total_emotions}\n")
            f.write(f"- Total de detecções faciais: {total_faces}\n")

    def process_video(self, input_path, output_path, known_face_encodings=None, known_face_names=None):
        """Process video for face recognition, emotion detection, and activity recognition."""
        print("Processing video...")
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {input_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        current_action = "Unknown"
        current_confidence = 0.0
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
        

                # Activity Recognition
                action, confidence = self.process_activity(frame)
                action, confidence = self.smooth_predictions(action, confidence)
                
                ANOMALY_THRESHOLD = 0.3
                
                # Exemplo de critério básico: se confiança < 0.3, consideramos anomalia
                if confidence < self.anomaly_threshold:
                    self.anomaly_counter += 1
                    # Opcional: desenhar no frame
                    cv2.putText(
                        frame, 
                        "Anomalia detectada", 
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 255), 
                        2
                    )
                    # Aqui, se quiser, não atualiza self.activity_counter
                else:
                    # Atualiza contador de atividade para atividades "normais"
                    current_action = action
                    current_confidence = confidence
                    self.activity_counter[current_action] = self.activity_counter.get(current_action, 0) + 1
                
                # Face Detection and Emotion Recognition
                aligned_faces, face_locations = self.detect_and_align_face(frame)
                
                for face, location in zip(aligned_faces, face_locations):
                    x1, y1, x2, y2 = location
                    
                    # Detectar emoção e atualizar contador
                    emotion = self.analyze_emotions(frame, location)
                    if emotion:
                        self.emotion_counter[emotion] = self.emotion_counter.get(emotion, 0) + 1
                    
                    # Extrair embedding e checar face conhecida
                    face_embedding = self.get_face_embedding(face)
                    name = "Nao Identificado"
                    
                    if known_face_encodings and known_face_names:
                        face_distances = [cosine(face_embedding, enc) for enc in known_face_encodings]
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] < 0.3:
                            name = known_face_names[best_match_index]
                            self.face_counter[name] = self.face_counter.get(name, 0) + 1
                    
                    # Desenhar bounding box e texto
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    emotion_translations = {
                        'happy': 'feliz', 'sad': 'triste', 'angry': 'irritado',
                        'fear': 'medo', 'surprise': 'surpreso', 'neutral': 'neutro',
                        'disgust': 'nojo'
                    }
                    if emotion:
                        emotion = emotion_translations.get(emotion.lower(), emotion)
                    
                    label = f"{name} | Emocao: {emotion}" if emotion else name
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (36, 255, 12), 
                        2
                    )
                
                # Renderizar texto da atividade e confiança
                # (caso queira exibir no frame se não for anomalia)
                self.draw_activity_results(frame, current_action, current_confidence)
                
                # Salvar frame processado
                out.write(frame)
                pbar.update(1)

            
        # Generate summary after processing
        summary_path = os.path.splitext(output_path)[0] + '_resumo.md'
        self.generate_summary(summary_path, total_frames)
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("\nVideo processing completed!")

    def draw_activity_results(self, frame, action, confidence):
        """Draw activity recognition results on frame."""
        height = frame.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (int(255 * (1 - confidence)), int(255 * confidence), 0)
        
        cv2.putText(frame, f"Atividade: {action}", (20, height - 60),
                   font, 1, color, 2)
        cv2.putText(frame, f"Confianca: {confidence:.2f}", (20, height - 20),
                   font, 1, color, 2)

def download_models():
    """Download required pre-trained models."""
    models = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "openface.nn4.small2.v1.t7": "http://openface-models.storage.cmusatyalab.org/nn4.small2.v1.t7"
    }
    
    print("Downloading required models...")
    for filename, url in models.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"{filename} downloaded successfully!")
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                raise

def load_known_faces(images_folder):
    """Load and encode known faces from images folder."""
    known_face_encodings = []
    known_face_names = []
    
    print("\nLoading known faces from images folder...")
    
    try:
        if not os.path.exists(images_folder):
            print(f"Warning: Images folder '{images_folder}' not found!")
            return [], []
            
        for filename in os.listdir(images_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_folder, filename)
                
                name = os.path.splitext(filename)[0]
                name = ''.join([i for i in name if not i.isdigit()])
                
                print(f"Processing {filename}...")
                image = cv2.imread(image_path)
                
                recognizer = CombinedVideoAnalyzer()
                faces, locations = recognizer.detect_and_align_face(image)
                
                if faces:
                    face_encoding = recognizer.get_face_embedding(faces[0])
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)
                    print(f"Successfully encoded face for {name}")
                else:
                    print(f"No face detected in {filename}")
                    
        print(f"\nLoaded {len(known_face_encodings)} known faces")
        return known_face_encodings, known_face_names
        
    except Exception as e:
        print(f"Error loading known faces: {str(e)}")
        return [], []

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video = os.path.join(script_dir, 'video.mp4')
    output_video = os.path.join(script_dir, 'output_combined.mp4')
    images_folder = os.path.join(script_dir, 'images')
    
    try:
        download_models()
        known_face_encodings, known_face_names = load_known_faces(images_folder)
        
        analyzer = CombinedVideoAnalyzer()
        analyzer.process_video(
            input_video,
            output_video,
            known_face_encodings,
            known_face_names
        )
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")

if __name__ == "__main__":
    main()
