############# refactor BY IA ############## 


from flask import Flask
from flask_restx import Api, Resource, fields, reqparse
from model import predict_genre

app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='FilmGenrAI', #Name create by IA 
    description='This tool predicts the genre of films'
)

ns = api.namespace('predict', description='Genre Predictor por films')

# Parser para los argumentos de entrada
parser = reqparse.RequestParser()
parser.add_argument(
    'plot',
    type=str,
    required=True,
    help='analyze',
    location='args'
)

# Definición de los campos de respuesta
resource_fields = api.model('PredictionResult', {
    'result': fields.Raw  # Porque estamos devolviendo un diccionario (dataframe convertido)
})


@ns.route('/')
class GenrePrediction(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        plot = args['plot']
        # Asegurar que plot sea texto
            if isinstance(plot, dict):
                plot = plot.get('plot', '')
            else:
                plot = str(plot)
        result_df = predict_genre(plot)
        return {'result': result_df.to_dict()}, 200


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
