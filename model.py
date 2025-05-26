#############Este archivo fue refactorizado con GPT############## 

import pandas as pd
import joblib
import neattext.functions as nfx
import neattext as nt

# Función de transformación y predicción
def predict_genre(plot: str) -> pd.DataFrame:
    
    script_dir = Path(__file__).parent

    # Construye las rutas completas a los archivos
    tfidf_path = script_dir / 'featureX.pkl'
    clf_path = script_dir / 'genreclf.pkl'

    # Carga los modelos
    tfidf = joblib.load(tfidf_path)
    clf = joblib.load(clf_path)

    plot_df = pd.DataFrame([plot], columns=['plot'])

    # Preprocesamiento del texto
    plot_df['plot'] = plot_df['plot'].apply(lambda x: nt.TextFrame(x).noise_scan())
    plot_df['plot'] = plot_df['plot'].apply(lambda x: nt.TextExtractor(x).extract_stopwords())
    plot_df['plot'] = plot_df['plot'].apply(nfx.remove_stopwords)

    Xfeatures = tfidf.transform(plot_df['plot']).toarray()

    # Predicción
    probabilities = clf.predict_proba(Xfeatures)

    cols = [
        'p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama',
        'p_Family', 'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News',
        'p_Romance', 'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western'
    ]

    genre_df = pd.DataFrame(probabilities, columns=cols)
    return genre_df.transpose().rename(columns={0: 'Proba'})
