import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf

from scripts.preprocess import preprocess_file

class MusicFileExplorer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MusicaGen - File Explorer")
        self.geometry("400x200")

        self.label = tk.Label(self, text="Selecione uma música para preprocessamento e geração", padx=10, pady=10)
        self.label.pack()

        self.select_button = tk.Button(self, text="Selecionar Música", command=self.select_file)
        self.select_button.pack(pady=20)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            preprocess_file(file_path)
            messagebox.showinfo("Sucesso", "Arquivo preprocessado com sucesso!")
            output_dir = self.create_generated_directory(file_path)
            self.generate_new_music(output_dir)

    def create_generated_directory(self, file_path):
        input_dir = os.path.dirname(file_path)
        output_dir = os.path.join(input_dir, 'generated')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def generate_new_music(self, output_dir):
        seed = np.random.rand(1, 13, 130)  # Exemplo de seed, ajuste conforme necessário
        generated_music = self.call_generate_script(seed)
        output_path = os.path.join(output_dir, 'generated_music.wav')
        sf.write(output_path, generated_music, 44100)  # Ajuste a taxa de amostragem conforme necessário
        messagebox.showinfo("Sucesso", f"Música gerada salva em: {output_path}")

    def call_generate_script(self, seed):
        model_url = 'https://tfhub.dev/google/magenta/music_vae/drums_2bar_lstm/2'
        model = hub.load(model_url)
        
        # A função de geração vai depender da entrada esperada e da saída do modelo
        # O modelo melody_rnn do Magenta não aceita uma seed diretamente dessa forma
        # É necessário construir um formato adequado para o modelo
        
        # Exemplificando como poderia ser feito com um formato de entrada adequado
        # Exemplo fictício - você precisa adaptar para o formato esperado pelo modelo real
        seed = tf.convert_to_tensor(seed, dtype=tf.float32)
        generated_music = model(seed)
        
        # Supondo que o modelo retorne um formato que possa ser convertido em áudio
        generated_music_np = generated_music.numpy()
        
        # Adaptar para o formato de áudio correto (exemplo fictício)
        return np.array(generated_music_np).reshape(-1, 1)

if __name__ == "__main__":
    app = MusicFileExplorer()
    app.mainloop()
