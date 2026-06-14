import dspy
from config import setup_dspy

setup_dspy()

def search_order(order_id : str):
    print(f"[TOOL] Consultando ordem no banco de dados: {order_id}")

    db = {
        "PED_001": {
            "status": "Entregue",
            "cliente": "João Silva",
            "produto": "Notebook Dell",
            "data_pedido": "2025-01-10",
            "data_entrega": "2025-01-15"
        },
        "PED_002": {
            "status": "Em transporte",
            "cliente": "Maria Souza",
            "produto": "iPhone 15",
            "data_pedido": "2025-01-18",
            "previsao_entrega": "2025-01-22"
        },
        "PED_003": {
            "status": "Processando",
            "cliente": "Carlos Oliveira",
            "produto": "Monitor LG 27",
            "data_pedido": "2025-01-20"
        },
        "PED_004": {
            "status": "Cancelado",
            "cliente": "Ana Costa",
            "produto": "Teclado Mecânico",
            "data_pedido": "2025-01-12"
        },
        "PED_005": {
            "status": "Aguardando pagamento",
            "cliente": "Pedro Santos",
            "produto": "PlayStation 5",
            "data_pedido": "2025-01-21"
        }
    }

    status = db.get(order_id, "Pedido não encontrado")
    print(f"[TOOL] Status encontrado: {status}")
    return status

class AgentSigature(dspy.Signature):
    query: str = dspy.InputField(desc = 'A dúvida do cliente')
    response: str = dspy.OutputField(desc = 'Uma resposta clara e bem educada')

class CostumerAgent(dspy.Module):
    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self.agent = dspy.ReAct(AgentSigature, tools=[search_order])
    
    def forward(self, query):
        return self.agent(query = query)

if __name__ == "__main__":
    agent = CostumerAgent()
    query = 'Onde está o meu pedido PED_001 e PED_002?'
    print("-" * 10 + " Iniciando atendimento " + "-" * 10)
    print(f'Dúvida do cliente: {query}')

    result = agent(query)

    print("-" * 10 + " Resumo da execução " + "-" * 10)
    print(f'Resposta final: {result.response}')
    print(f'Raciocínio: {result}')

   




