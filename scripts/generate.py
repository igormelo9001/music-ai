import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import soundfile as sf

def generate_music(model_url, seed):
    # Carregar o modelo do TensorFlow Hub
    model = hub.load(model_url)
    
    # Gerar música
    generated_music = model(seed)
    
    return generated_music.numpy()

if __name__ == "__main__":
    # URL do modelo
    model_url = 'https://tfhub.dev/google/magenta/melody_rnn/2'
    
    # Exemplo de seed
    seed = tf.random.normal([1, 128])
    
    # Gerar música
    generated_music = generate_music(model_url, seed)
    
    # Salvar música gerada
    output_path = 'generated_music.wav'
    sf.write(output_path, generated_music, 44100)  # Ajuste a taxa de amostragem conforme necessário
    print(f"Generated music saved to {output_path}")
