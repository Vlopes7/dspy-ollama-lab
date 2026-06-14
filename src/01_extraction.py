import dspy
import json
from pydantic import BaseModel, Field
from config import setup_dspy

setup_dspy()

class TicketAnalysis(BaseModel):
    name: str = Field(description='Nome do cliente')
    issue: str = Field(description='Problema principal relatado')
    urgency : int = Field(description='Escala de urgência de 1 a 10')
    sentiment: str = Field(description='Sentimento: Positivo, Neutro ou Negativo')

class ExtractTicket(dspy.Signature):
    email: str = dspy.InputField(desc = "O texto completo do e-mail")
    analysis: TicketAnalysis = dspy.OutputField(desc = 'Análise estruturada do ticker')

if __name__ == "__main__":
    with open('../data/support_emails.json', 'r', encoding='utf-8') as f:
        emails = json.load(f)

    predict_module = dspy.Predict(ExtractTicket)
    cot_module = dspy.ChainOfThought(ExtractTicket)

    print('-' * 20)
    print('Comparativo: predict vs throught')
    print('-' * 20)

    for item in emails:
        content = item['content']
        print(f"ID: {item['id']}")
        print(f"Content: {item['content']}")
        print('-' * 20)
    
        res_predict = predict_module(email = content)
        print(f"[PREDICT] Urgência: {res_predict.analysis.urgency} | Sentimento: {res_predict.analysis.sentiment}")

        res_cot = cot_module(email = content)
        print(f"[CoT] Urgência {res_cot.analysis.urgency} | Sentimento: {res_cot.analysis.sentiment}")

        print(f"Raciocínio: {res_cot.reasoning}")


