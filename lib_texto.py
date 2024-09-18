import pandas as pd
import nltk 
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt 


text="Texto para poner a prueba la instalación de librerias"
tokens = word_tokenize(text)
print("Tokens: ", tokens)

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
print("Lematizados:", lemmatized_words)

word_freq = pd.Series(lemmatized_words).value_counts()
print("Frecuencia de palabras:\n", word_freq)

plt.plot(word_freq)
plt.xlabel("Palabras")
plt.ylabel("Frecuencia")
plt.title("Frecuencia de palabras")
plt.show()