from .config.config import config
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


def format_web_results(web_result: dict) -> str:
    """
    Converte o resultado bruto da Tavily em texto para o prompt.
    """
    parts = []

    answer = web_result.get("answer")
    if answer:
        parts.append(f"RESUMO DA WEB:\n{answer}")

    results = web_result.get("results", [])
    if results:
        parts.append("\nFONTES ENCONTRADAS:")
        for i, item in enumerate(results, start=1):
            title = item.get("title", "Sem título")
            url = item.get("url", "Sem URL")
            content = item.get("content", "")
            parts.append(
                f"\n[{i}] {title}\nURL: {url}\nTrecho: {content}"
            )

    return "\n".join(parts).strip()


def build_prompt(user_question: str, web_context: str) -> str:
    return f"""
Você é um analista financeiro especializado em investimentos no Brasil.

Considere:
- produtos disponíveis para investidores brasileiros
- Tesouro Direto
- ETFs da B3
- FIIs
- ações brasileiras
Responda sempre em português do Brasil.

Sua tarefa:
1. Ler a pergunta do usuário.
2. Usar o contexto da web fornecido abaixo.
3. Produzir uma resposta prática, organizada e prudente.
4. Quando houver incerteza macroeconômica ou geopolítica, destacar riscos.
5. Não inventar fatos que não estejam no contexto.
6. No final, incluir um aviso breve de que a resposta não substitui aconselhamento financeiro profissional.

Pergunta do usuário:
{user_question}

Contexto obtido da web:
{web_context}

Formato desejado da resposta:
- Cenário resumido
- Proposta de carteira para R$ 100.000
- Justificativa de cada bloco
- Principais riscos
- Conclusão objetiva
""".strip()


def responder_pergunta(user_question: str) -> str:
    model_name = config["models"]["chat_model"]

    search_cfg = config.get("search", {})
    max_results = search_cfg.get("max_results", 5)
    topic = search_cfg.get("topic", "finance")
    search_depth = search_cfg.get("search_depth", "advanced")

    llm = ChatOllama(
        model=model_name,
        temperature=0
    )

    web_search = TavilySearch(
        max_results=max_results,
        topic=topic,
        search_depth=search_depth,
        include_answer=True
    )

    web_result = web_search.invoke({"query": user_question})
    web_context = format_web_results(web_result)
    prompt = build_prompt(user_question, web_context)

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    
    pergunta = (
    """
Você é um analista financeiro experiente.

Preciso de uma recomendação de investimento de R$ 100.000 para um perfil moderado a conservador, considerando o atual cenário de guerra entre Israel, EUA e Irã.

Sua resposta deve seguir esta estrutura:
1. Contexto do investidor (levando em conta meu perfil)
2. Alocação comentada (percentuais e justificativa)
3. Veículos específicos para cada classe (ex.: ações de setores, tipos de renda fixa, ouro, petróleo)
4. Principais riscos e como a carteira ajuda a mitigá-los
5. Recomendação de monitoramento

Não use linguagem metalinguística como “Uma resposta mais robusta incluiria:”. Responda diretamente ao usuário.
"""
    )

    resposta = responder_pergunta(pergunta)
    print("\n=== RESPOSTA FINAL ===\n")
    print(resposta)