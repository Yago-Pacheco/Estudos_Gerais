from fastapi import FastAPI

# Criando um instância do objeto FastAPI
app = FastAPI()


@app.get('/')
def read_root():
    return {'message': 'Olá mundo'}
