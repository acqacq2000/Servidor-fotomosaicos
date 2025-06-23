import eventlet  # Importa Eventlet, una biblioteca para concurrencia asíncrona compatible con Flask-SocketIO

# Aplica monkey patching para que Eventlet funcione correctamente con otras bibliotecas
eventlet.monkey_patch()

from flask import Flask, request, render_template, send_from_directory, send_file  # Importa componentes clave de Flask para el servidor web
import os  # Permite operaciones con el sistema de archivos
import cv2  # OpenCV para procesamiento de imágenes
import numpy as np  # Biblioteca para operaciones matemáticas y de matrices
import random  # Biblioteca estándar para generar valores aleatorios
from PIL import Image, ImageOps, ImageDraw, ImageFont  # PIL para manipulación avanzada de imágenes
from tqdm import tqdm  # Muestra barra de progreso en bucles
from math import ceil  # Importa función para redondear hacia arriba
from tensorflow.keras.models import Model  # Clase base para construir modelos Keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, LeakyReLU  # Capas de red neuronal
from tensorflow.keras.optimizers import Adam  # Optimizador de red neuronal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Callbacks para entrenamiento
from sklearn.metrics.pairwise import cosine_distances  # Métrica para calcular distancia entre vectores
from io import BytesIO  # Permite manipular archivos en memoria como si fueran archivos físicos
import base64  # Para codificar y decodificar datos en Base64
from flask_socketio import SocketIO, emit  # Para comunicación WebSocket con clientes en tiempo real
from threading import Thread  # Permite crear y controlar hilos de ejecución
from tensorflow.keras.callbacks import Callback  # Clase base para crear callbacks personalizados en Keras
import time  # Para manejo de tiempo
import eventlet  # Reimportado accidentalmente, innecesario si ya se importó arriba

# Clase personalizada que fuerza un pequeño "descanso" para que Eventlet procese otros eventos mientras se entrena el modelo
class KeepAliveCallback(Callback):
    def __init__(self, intervalo=5):  # Constructor que define el intervalo de espera entre lotes
        super().__init__()  # Inicializa la clase base Callback
        self.last_yield = time.time()  # Guarda el tiempo actual
        self.intervalo = intervalo  # Define intervalo en segundos para ceder el control

    def on_batch_end(self, batch, logs=None):  # Método que se ejecuta después de cada batch de entrenamiento
        ahora = time.time()  # Registra el tiempo actual
        if ahora - self.last_yield > self.intervalo:  # Si ha pasado más del intervalo definido
            eventlet.sleep(0)  # Permite al servidor procesar otros eventos
            self.last_yield = ahora  # Actualiza el último tiempo registrado

# Configuración de parámetros del mosaico
TILE_SIZE = 128  # Tamaño de cada tile del mosaico (en píxeles)
DESIRED_NUM_BLOCKS = 4250  # Número deseado de bloques en el mosaico final
MAX_USAGE_RATIO = 2.0  # Máximo de veces que puede repetirse un tile
UPLOAD_FOLDER = 'uploads'  # Carpeta donde se guardan las imágenes objetivo cargadas
OUTPUT_FOLDER = 'outputs'  # Carpeta para guardar el resultado final del mosaico
TILE_FOLDER = 'tiles_folder'  # Carpeta para guardar los tiles procesados

# Crea las carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Crea carpeta para imágenes objetivo si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Crea carpeta para resultados si no existe
os.makedirs(TILE_FOLDER, exist_ok=True)  # Crea carpeta para tiles si no existe

# Inicializa la aplicación Flask y SocketIO
app = Flask(__name__)  # Crea instancia de la app Flask
socketio = SocketIO(app)  # Inicializa SocketIO con Flask para comunicación en tiempo real
connected_clients = 0  # Contador de clientes conectados por WebSocket

# --- Definición del modelo Autoencoder ---
def build_autoencoder():
    inp = Input(shape=(TILE_SIZE, TILE_SIZE, 1))  # Capa de entrada, imagen en escala de grises con tamaño TILE_SIZE x TILE_SIZE
    x = Conv2D(64, 3, padding='same')(inp)  # Primera capa convolucional con 64 filtros
    x = BatchNormalization()(x)  # Normaliza la salida para estabilizar entrenamiento
    x = LeakyReLU()(x)  # Activación LeakyReLU
    x = MaxPooling2D(2, padding='same')(x)  # Reduce tamaño a la mitad conservando información espacial

    x = Conv2D(128, 3, padding='same')(x)  # Segunda capa convolucional
    x = BatchNormalization()(x)  # Normalización
    x = LeakyReLU()(x)  # Activación LeakyReLU
    x = MaxPooling2D(2, padding='same')(x)  # Otro downsampling

    x = Dropout(0.3)(x)  # Dropout para evitar sobreajuste
    x = Conv2D(256, 3, padding='same')(x)  # Capa con mayor profundidad
    x = BatchNormalization()(x)  # Normalización
    encoded = LeakyReLU()(x)  # Representación codificada (latente)

    x = UpSampling2D(2)(encoded)  # Empieza decodificación (aumenta resolución)
    x = Conv2D(128, 3, padding='same')(x)  # Capa deconvolucional
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = UpSampling2D(2)(x)  # Aumenta resolución nuevamente
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(x)  # Reconstrucción final con activación sigmoide

    autoencoder = Model(inp, decoded)  # Modelo completo autoencoder
    encoder = Model(inp, encoded)  # Solo la parte del codificador
    autoencoder.compile(optimizer=Adam(1e-4), loss='mse')  # Compila el modelo con MSE y Adam optimizador
    return autoencoder, encoder  # Devuelve ambos modelos

# --- Preprocesamiento de imágenes tiles ---
def preparar_tiles(input_dir, output_dir, desired_size=128):
    os.makedirs(output_dir, exist_ok=True)  # Asegura que el directorio de salida exista
    for file_name in os.listdir(input_dir):  # Itera sobre los archivos del directorio de entrada
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filtra archivos de imagen
            input_path = os.path.join(input_dir, file_name)  # Ruta completa del archivo de entrada
            output_path = os.path.join(output_dir, file_name)  # Ruta donde se guardará la imagen procesada
            with Image.open(input_path) as img:  # Abre la imagen
                gray_img = img.convert("L")  # Convierte la imagen a escala de grises
                square_img = ImageOps.fit(gray_img, (desired_size, desired_size), Image.Resampling.LANCZOS)  # Redimensiona manteniendo aspecto
                square_img.save(output_path)  # Guarda la imagen procesada

# --- Preprocesamiento del objetivo (imagen principal) ---
def preparar_target(input_path, output_path):
    with Image.open(input_path) as img:  # Abre la imagen original
        gray_img = img.convert("L")  # Convierte a escala de grises
        np_gray = np.array(gray_img)  # Convierte a array de NumPy
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Inicializa algoritmo CLAHE para mejorar contraste
        enhanced = clahe.apply(np_gray)  # Aplica CLAHE sobre la imagen
        Image.fromarray(enhanced).save(output_path)  # Guarda la imagen mejorada

# --- Mejora del contraste de las tiles ---
def stretch_soft_bimodal(tile, target_min=0, target_max=200):
    tile = tile.astype(np.float32)  # Convierte la imagen a tipo float
    tile_min = tile.min()  # Mínimo valor de intensidad
    tile_max = tile.max()  # Máximo valor de intensidad
    if tile_max == tile_min:  # Si todos los píxeles son iguales
        return np.full_like(tile, (target_min + target_max) / 2 / 255.0)  # Devuelve imagen uniforme
    center = (tile_min + tile_max) / 2.0  # Calcula el centro del histograma
    correction = np.zeros_like(tile)  # Inicializa imagen de corrección

    low_mask = tile < center  # Máscara para la mitad inferior
    correction[low_mask] = (tile[low_mask] - tile_min) / (center - tile_min + 1e-5)  # Normaliza inferior
    correction[low_mask] = target_min + correction[low_mask] * (center - target_min)  # Ajusta rango inferior

    high_mask = tile >= center  # Máscara para la mitad superior
    correction[high_mask] = (tile[high_mask] - center) / (tile_max - center + 1e-5)  # Normaliza superior
    correction[high_mask] = center + correction[high_mask] * (target_max - center)  # Ajusta rango superior

    corrected = np.clip(correction, 0, 255) / 255.0  # Normaliza a [0,1] y limita valores extremos
    return corrected  # Devuelve imagen ajustada

# --- Carga las imágenes en escala de grises desde un folder ---
def load_gray_tiles(folder):
    tiles = []  # Lista para almacenar las tiles cargadas
    for fname in sorted(os.listdir(folder)):  # Itera sobre los archivos ordenados del directorio
        path = os.path.join(folder, fname)  # Ruta completa del archivo
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Lee la imagen en modo escala de grises
        if img is not None:  # Verifica que la imagen se haya cargado correctamente
            resized = cv2.resize(img, (TILE_SIZE, TILE_SIZE))  # Redimensiona la imagen a TILE_SIZE x TILE_SIZE
            tiles.append(resized)  # Añade la imagen a la lista
    return np.array(tiles)  # Devuelve un array de NumPy con las imágenes

# --- Divide la imagen objetivo en bloques (tiles virtuales) ---
def split_blocks(img):
    h, w = img.shape  # Obtiene altura y anchura de la imagen
    blocks = []  # Lista donde se guardarán los bloques
    for y in range(0, h, TILE_SIZE):  # Recorre la imagen verticalmente por pasos de TILE_SIZE
        for x in range(0, w, TILE_SIZE):  # Recorre la imagen horizontalmente por pasos de TILE_SIZE
            tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]  # Extrae un bloque de la imagen
            if tile.shape == (TILE_SIZE, TILE_SIZE):  # Verifica que el bloque tenga el tamaño adecuado
                blocks.append(((y, x), tile))  # Añade una tupla con la posición y el bloque
    return blocks  # Devuelve la lista de bloques

# --- Construcción eficiente del mosaico ---
def build_mosaic_fast(blocks, block_feats, tile_features, tile_images, original_img, alpha_blend=0.2):
    h_blocks = max(y for (y, _), _ in blocks) // TILE_SIZE + 1  # Número total de bloques verticales
    w_blocks = max(x for (_, x), _ in blocks) // TILE_SIZE + 1  # Número total de bloques horizontales
    num_blocks = len(blocks)  # Total de bloques en la imagen objetivo
    num_tiles = len(tile_images)  # Total de tiles disponibles
    avg_usage = ceil(num_blocks / num_tiles)  # Uso promedio permitido por tile
    max_usage = int(avg_usage * MAX_USAGE_RATIO)  # Uso máximo permitido por tile

    usage = np.zeros(num_tiles, dtype=int)  # Contador de usos por cada tile
    mosaic = np.zeros((h_blocks * TILE_SIZE, w_blocks * TILE_SIZE), dtype=np.float32)  # Imagen del mosaico final
    placed_tiles = -np.ones((h_blocks, w_blocks), dtype=int)  # Matriz para llevar registro de las tiles colocadas

    block_data = list(zip(blocks, block_feats))  # Combina posición y características de cada bloque
    random.shuffle(block_data)  # Mezcla aleatoriamente los bloques para evitar patrones visibles

    socketio.emit('mosaic_init', {'cols': w_blocks, 'rows': h_blocks})  # Envía información inicial al cliente

    for ((y_pos, x_pos), block_img), block_feat in tqdm(block_data):  # Itera sobre cada bloque con barra de progreso
        dists = cosine_distances([block_feat], tile_features)[0]  # Calcula distancias coseno con todas las tiles
        penalized = dists + (usage / max_usage) * 0.3  # Penaliza tiles ya muy usadas
        sorted_idx = np.argsort(penalized)  # Ordena por menor distancia penalizada

        h_idx = y_pos // TILE_SIZE  # Índice vertical del bloque
        w_idx = x_pos // TILE_SIZE  # Índice horizontal del bloque

        # Función para evitar colocar tiles repetidas en posiciones adyacentes
        def is_conflicting(candidate_idx):
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = h_idx + dy, w_idx + dx
                    if 0 <= ny < h_blocks and 0 <= nx < w_blocks:
                        if placed_tiles[ny][nx] == candidate_idx:
                            return True
            return False

        for idx in sorted_idx:  # Busca la mejor tile que no se haya usado demasiado ni esté repetida cerca
            if usage[idx] < max_usage and not is_conflicting(idx):
                best_idx = idx
                break
        else:
            best_idx = sorted_idx[0]  # Si no se encuentra una válida, toma la de menor distancia

        placed_tiles[h_idx][w_idx] = best_idx  # Marca la tile como colocada en esa posición
        usage[best_idx] += 1  # Incrementa su uso

        tile = tile_images[best_idx].squeeze()  # Obtiene la imagen de la tile
        block = block_img.astype(np.float32) / 255.0  # Normaliza el bloque original
        mean_block = block.mean()  # Promedio de brillo del bloque
        mean_tile = tile.mean()  # Promedio de brillo de la tile
        brightness_adjustment = (mean_block - mean_tile) * 0.1  # Calcula ajuste de brillo

        tile = tile + brightness_adjustment  # Ajusta el brillo de la tile
        tile = np.clip(tile, 0.0, 1.0)  # Limita valores a rango válido
        tile_mixed = (tile * (1 - alpha_blend)) + (block * alpha_blend)  # Mezcla tile con el bloque original
        tile_mixed = stretch_soft_bimodal(tile_mixed * 255)  # Mejora el contraste

        thumb = tile_to_base64(tile_mixed)  # Convierte tile a base64 para previsualización
        socketio.emit('tile_placed', {'col': w_idx, 'row': h_idx, 'thumb': thumb})  # Envía al cliente la tile colocada

        eventlet.sleep(0.005)  # Permite que otros eventos del servidor se procesen

        mosaic[y_pos:y_pos+TILE_SIZE, x_pos:x_pos+TILE_SIZE] = tile_mixed  # Coloca la tile en el mosaico

    final = (mosaic * 255).clip(0, 255).astype(np.uint8)  # Convierte la imagen final a formato de 8 bits
    return final  # Devuelve el mosaico completo

# --- Pipeline principal que construye el mosaico completo ---
def crear_fotomosaico(ruta_objetivo, carpeta_tiles):
    tiles_preparadas = TILE_FOLDER  # Ruta de carpeta temporal de tiles procesadas
    ruta_target_procesada = os.path.join(OUTPUT_FOLDER, 'target_gray.jpg')  # Imagen objetivo preprocesada
    ruta_final = os.path.join(OUTPUT_FOLDER, 'fotomosaico.jpg')  # Ruta donde se guardará el mosaico final

    preparar_tiles(carpeta_tiles, tiles_preparadas)  # Procesa y normaliza las tiles
    preparar_target(ruta_objetivo, ruta_target_procesada)  # Convierte la imagen objetivo a escala de grises y mejora contraste

    target = cv2.imread(ruta_target_procesada, cv2.IMREAD_GRAYSCALE)  # Carga imagen objetivo como array en escala de grises
    h, w = target.shape  # Obtiene dimensiones de la imagen objetivo
    aspect_ratio = w / h  # Calcula relación de aspecto (ancho/alto)

    num_blocks_h = int(np.sqrt(DESIRED_NUM_BLOCKS / aspect_ratio))  # Número de bloques verticales
    num_blocks_w = int(num_blocks_h * aspect_ratio)  # Número de bloques horizontales ajustado al aspecto
    target = cv2.resize(target, (num_blocks_w * TILE_SIZE, num_blocks_h * TILE_SIZE))  # Redimensiona imagen para que encaje con tiles

    blocks = split_blocks(target)  # Divide la imagen en bloques de tamaño TILE_SIZE
    tile_imgs = load_gray_tiles(tiles_preparadas)  # Carga todas las tiles en escala de grises

    if tile_imgs.shape[0] == 0:
        raise ValueError("No se han encontrado tiles válidas en la carpeta.")  # Verifica que haya tiles válidas

    tile_imgs = tile_imgs.astype(np.float32) / 255.0  # Normaliza valores de píxeles entre 0 y 1
    tile_imgs = tile_imgs[..., np.newaxis]  # Añade dimensión para canales (1 canal - escala de grises)

    autoencoder, encoder = build_autoencoder()  # Construye modelo autoencoder y extrae el codificador (encoder)

    autoencoder.fit(
        tile_imgs, tile_imgs,
        epochs=300,
        batch_size=128,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),  # Detiene si la pérdida no mejora
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6),  # Reduce LR si estancado
            KeepAliveCallback(intervalo=5)  # Mantiene el servidor activo durante entrenamiento
        ]
    )

    tile_feats = encoder.predict(tile_imgs, batch_size=128).reshape(len(tile_imgs), -1)  # Extrae características de tiles
    tile_feats /= np.linalg.norm(tile_feats, axis=1, keepdims=True) + 1e-8  # Normaliza vectores

    block_imgs = np.array([b for (_, b) in blocks], dtype=np.float32) / 255.0  # Normaliza bloques de imagen
    block_imgs = block_imgs[..., np.newaxis]  # Añade dimensión de canal
    block_feats = encoder.predict(block_imgs, batch_size=128).reshape(len(blocks), -1)  # Extrae características de bloques
    block_feats /= np.linalg.norm(block_feats, axis=1, keepdims=True) + 1e-8  # Normaliza vectores

    mosaic = build_mosaic_fast(blocks, block_feats, tile_feats, tile_imgs, target)  # Construye el mosaico

    cv2.imwrite(ruta_final, mosaic)  # Guarda la imagen final en disco
    socketio.emit('mosaic_done', {'status': 'success', 'result_url': '/outputs/fotomosaico.jpg'})  # Informa al cliente
    return ruta_final  # Devuelve la ruta del mosaico generado

# --- Rutas Flask para la aplicación web ---
@app.route('/')
def index():
    return render_template('index.html')  # Devuelve la página principal de la aplicación

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)  # Sirve archivos desde la carpeta de resultados

# --- Manejo de conexiones con WebSocket ---
@socketio.on('connect')
def on_connect():
    global connected_clients
    connected_clients += 1  # Incrementa el contador de clientes conectados
    print(f'[✅] Cliente conectado. Total: {connected_clients}')

@socketio.on('disconnect')
def on_disconnect():
    global connected_clients
    connected_clients = max(0, connected_clients - 1)  # Reduce el contador de clientes conectados
    print(f'[🔴] Cliente desconectado. Total: {connected_clients}')

@socketio.on('keepalive')
def handle_keepalive():
    print("[🔁] Keepalive recibido del cliente")  # Confirma señal de actividad del cliente
    emit('pong')  # Responde con un ping

@socketio.on('start')
def on_start():
    # Simulación para testeo (envía tiles de colores al cliente)
    cols = 10  # Número de columnas en la grilla simulada
    colores = ["red", "green", "blue", "yellow", "orange",
               "purple", "cyan", "magenta", "lime", "pink"]
    for i in range(100):
        fila = i // cols
        col = i % cols
        x = col * 32
        y = fila * 32
        color = colores[i % len(colores)]
        base64_tile = generar_tile_base64(color, x, y)  # Genera tile de color en base64
        socketio.emit('tile_placed', {'x': x, 'y': y, 'thumb': base64_tile})  # Envía al cliente
        eventlet.sleep(0.1)  # Pausa breve entre envíos

# --- Conversión de array de imagen a string base64 para enviar por WebSocket ---
def tile_to_base64(tile_array):
    tile_image = Image.fromarray((tile_array * 255).astype(np.uint8))  # Convierte a imagen PIL
    buffer = BytesIO()
    tile_image.save(buffer, format="PNG")  # Guarda en buffer
    return base64.b64encode(buffer.getvalue()).decode('utf-8')  # Devuelve string codificado

# --- Ruta para subir imagen objetivo y tiles desde cliente ---
@app.route('/subir', methods=['POST'])
def subir():
    objetivo = request.files['foto']  # Imagen objetivo enviada por el usuario
    ruta_objetivo = os.path.join(UPLOAD_FOLDER, 'objetivo_' + objetivo.filename)
    objetivo.save(ruta_objetivo)  # Guarda la imagen

    img = Image.open(objetivo).convert("RGB")  # Abre la imagen y la convierte a RGB
    buffer = BytesIO()
    for quality in range(95, 10, -5):  # Intenta comprimir hasta quedar bajo 400KB
        buffer.seek(0)
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        size_kb = buffer.tell() / 1024
        if size_kb <= 400:
            break
    with open(ruta_objetivo, 'wb') as f:
        f.write(buffer.getvalue())  # Sobrescribe archivo comprimido

    # Limpia carpeta de tiles anteriores
    for f in os.listdir(TILE_FOLDER):
        os.remove(os.path.join(TILE_FOLDER, f))

    tile_files = request.files.getlist('tiles')  # Lista de tiles subidas por el usuario
    for i, tile in enumerate(tile_files):
        if tile and tile.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            nombre_tile = f"tile_{i}_{tile.filename}"
            ruta_guardado = os.path.join(TILE_FOLDER, nombre_tile)
            tile.save(ruta_guardado)  # Guarda cada tile

    if not os.listdir(TILE_FOLDER):  # Verifica que se hayan recibido tiles válidas
        return "Error: No se recibieron imágenes tiles válidas.", 400

    # Lanza la construcción del mosaico en segundo plano
    def tarea_mosaico():
        try:
            crear_fotomosaico(ruta_objetivo, TILE_FOLDER)
        except Exception as e:
            print(f"[❌] Error durante creación del mosaico: {e}")
            socketio.emit('mosaic_done', {'status': 'error', 'message': str(e)})

    socketio.start_background_task(tarea_mosaico)
    return "OK", 200  # Respuesta inmediata

# --- Inicia el servidor Flask con WebSocket ---
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)  # Inicia la app escuchando conexiones