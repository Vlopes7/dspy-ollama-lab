import dspy
from dspy.teleprompt import BootstrapFewShot
from config import setup_dspy

setup_dspy()

with open("../context/departments.txt", "r", encoding="utf-8") as f:
    DEPARTMENTS_CONTEXT = f.read()

class IntentSignature(dspy.Signature):
    """
    Classifique o email para o departamento correto.

    RH: férias, folha de pagamento, benefícios, admissões e cadastro.
    TI: sistemas, computadores, acessos, VPN, impressoras e suporte.
    JURÍDICO: contratos, processos, notificações e compliance.
    FINANCEIRO: pagamentos, cobranças, notas fiscais e reembolsos.
    COMPRAS: cotações, fornecedores e aquisições.
    LOGÍSTICA: estoque, transporte, expedição e entregas.
    COMERCIAL: propostas, negociações e oportunidades de negócio.
    ATENDIMENTO: suporte ao cliente, dúvidas e reclamações.
    """

    email: str = dspy.InputField(desc="Conteúdo do email recebido")
    context: str = dspy.InputField(desc="Função de cada setor")

    department: str = dspy.OutputField(
        desc="""
        Retorne APENAS um dos valores:

        RH
        TI
        JURÍDICO
        FINANCEIRO
        COMPRAS
        LOGÍSTICA
        COMERCIAL
        ATENDIMENTO
        """
    )

trainset = [
    dspy.Example(
        email="Gostaria de solicitar minhas férias para o período de dezembro.",
        context=DEPARTMENTS_CONTEXT,
        department="RH"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Recebi meu holerite com informações incorretas sobre horas extras.",
        context=DEPARTMENTS_CONTEXT,
        department="RH"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Preciso atualizar meus dados cadastrais e dependentes no sistema.",
        context=DEPARTMENTS_CONTEXT,
        department="RH"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Gostaria de obter informações sobre o plano de saúde corporativo.",
        context=DEPARTMENTS_CONTEXT,
        department="RH"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Não consigo acessar minha conta de e-mail desde esta manhã.",
        context=DEPARTMENTS_CONTEXT,
        department="TI"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Meu computador está apresentando lentidão e travamentos constantes.",
        context=DEPARTMENTS_CONTEXT,
        department="TI"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Preciso de acesso ao sistema ERP para executar minhas atividades.",
        context=DEPARTMENTS_CONTEXT,
        department="TI"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="A impressora do setor não está funcionando corretamente.",
        context=DEPARTMENTS_CONTEXT,
        department="TI"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Recebemos uma notificação judicial sobre o contrato de prestação de serviços.",
        context=DEPARTMENTS_CONTEXT,
        department="JURÍDICO"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Solicito análise jurídica da minuta de contrato enviada pelo fornecedor.",
        context=DEPARTMENTS_CONTEXT,
        department="JURÍDICO"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Precisamos revisar os termos de confidencialidade antes da assinatura.",
        context=DEPARTMENTS_CONTEXT,
        department="JURÍDICO"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Foi recebida uma reclamação formal de um cliente com possível implicação legal.",
        context=DEPARTMENTS_CONTEXT,
        department="JURÍDICO"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Identifiquei uma divergência no valor da nota fiscal recebida.",
        context=DEPARTMENTS_CONTEXT,
        department="FINANCEIRO"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Gostaria de verificar o status do pagamento da fatura enviada.",
        context=DEPARTMENTS_CONTEXT,
        department="FINANCEIRO"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Solicito reembolso das despesas de viagem realizadas na semana passada.",
        context=DEPARTMENTS_CONTEXT,
        department="FINANCEIRO"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="O fornecedor informou que ainda não recebeu o pagamento acordado.",
        context=DEPARTMENTS_CONTEXT,
        department="FINANCEIRO"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Precisamos adquirir novos notebooks para a equipe de desenvolvimento.",
        context=DEPARTMENTS_CONTEXT,
        department="COMPRAS"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Solicito cotação para aquisição de materiais de escritório.",
        context=DEPARTMENTS_CONTEXT,
        department="COMPRAS"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Gostaria de receber uma proposta comercial para contratação dos serviços.",
        context=DEPARTMENTS_CONTEXT,
        department="COMERCIAL"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Tenho interesse em conhecer os planos disponíveis para empresas.",
        context=DEPARTMENTS_CONTEXT,
        department="COMERCIAL"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Estou enfrentando dificuldades para utilizar a plataforma contratada.",
        context=DEPARTMENTS_CONTEXT,
        department="ATENDIMENTO"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Gostaria de registrar uma reclamação sobre o atendimento recebido.",
        context=DEPARTMENTS_CONTEXT,
        department="ATENDIMENTO"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="A entrega do pedido está atrasada.",
        context=DEPARTMENTS_CONTEXT,
        department="LOGÍSTICA"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="Precisamos rastrear uma carga enviada ao cliente.",
        context=DEPARTMENTS_CONTEXT,
        department="LOGÍSTICA"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="O estoque apresenta divergência de inventário.",
        context=DEPARTMENTS_CONTEXT,
        department="LOGÍSTICA"
    ).with_inputs("email", "context"),

    dspy.Example(
        email="O caminhão não chegou ao centro de distribuição.",
        context=DEPARTMENTS_CONTEXT,
        department="LOGÍSTICA"
    ).with_inputs("email", "context"),
]

def department_metric(gold, pred, trace=None):
    return gold.department.strip().upper() == pred.department.strip().upper()

if __name__ == "__main__":
    print("OTIMIZAÇÃO DE TRIAGEM CORPORATIVA")

    optimizer = BootstrapFewShot(
        metric=department_metric,
        max_bootstrapped_demos=3
    )

    student = dspy.ChainOfThought(IntentSignature)

    compiled_program = optimizer.compile(
        student=student,
        trainset=trainset
    )

    test_cases = [
        "A impressora do setor de vendas parou de funcionar e precisamos de um técnico",
        "Não consigo acessar a VPN da empresa para trabalhar remotamente.",
        "Recebemos uma intimação relacionada a um processo em andamento.",
        "A transportadora informou atraso na entrega da mercadoria.",
        "Desejo solicitar um orçamento para 200 licenças da plataforma.",
        "Gostaria de registrar uma reclamação sobre o suporte recebido.",
        "O caminhão responsável pela entrega não chegou ao centro de distribuição."
    ]

    print("RESULTADOS DA TRIAGEM OTIMIZADA:")

    for email in test_cases:
        result = compiled_program(
            email=email,
            context=DEPARTMENTS_CONTEXT
        )

        print(f"\nEmail: {email}")
        print(f"Destino: {result.department}")

        if hasattr(result, "reasoning"):
            print(f"Razão: {result.reasoning}")