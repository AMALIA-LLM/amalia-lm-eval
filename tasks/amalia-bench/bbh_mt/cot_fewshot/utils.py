import re


class ExtractAnswerR:
    """
    Extracts the answer from the response and returns it.
    """

    def __init__(self):
        pass

    def apply(self, resps: list[list[str]], docs: list[dict]):

        # print(resps)
        # print(docs)

        def extract_answer(responses):

            filtered_responses = []
            for r in responses:
                """Extract the answer letter from the model's response."""
                # Look for (a resposta| o argumento) (corret(o|a))? é:? \(A-Z\) something else
                # if A-Z use the results inside it else grab everything after é:? until the end of the line or a full stop or comma but not include the full stop or comma
                # also remove all leading and trailing spaces and '' and "" around the answer

                # try the simple one grab the letter and the parentheses
                answer_pattern = re.search(r"(a resposta|o argumento|o resultado) (correc?t[oa] |cert[oa] |final )?é:?\s*(\([A-Z]\))", r, re.IGNORECASE | re.MULTILINE)
                if answer_pattern:
                    answer = answer_pattern.group(3).strip()
                    filtered_responses.append(answer)
                    break

                # original regex
                answer_pattern = re.search(r"(a resposta|o argumento|o resultado) (correc?t[oa] |cert[oa] |final )?é:?(.*?)(?:[.\\s]*)$", r, re.IGNORECASE | re.MULTILINE)
                if answer_pattern:
                    answer = answer_pattern.group(3).strip().replace("'", "").replace('"', '').replace("`", "").replace("**", "").strip()
                    # if ends with dot remove it
                    if answer.endswith("."):
                        answer = answer[:-1].strip()

                    # special case to handle gender valida/falsa/verdadeira to valido/falso/verdadeiro with ignore case
                    answer = re.sub(r"\bválida\b", "Válido", answer, flags=re.IGNORECASE)
                    answer = re.sub(r"\binválida\b", "Inválido", answer, flags=re.IGNORECASE)

                    answer = re.sub(r"\bfalsa\b", "Falso", answer, flags=re.IGNORECASE)
                    answer = re.sub(r"\bverdadeira\b", "Verdadeiro", answer, flags=re.IGNORECASE)

                    filtered_responses.append(answer)
                    break

                # grab the last letter between parentheses if there are multiple instances and support multiline
                final_letter = re.findall(r"\(([A-Z])\)", r.strip(), re.MULTILINE)
                if final_letter:
                    filtered_responses.append("(" + final_letter[-1] + ")")
                    break

                filtered_responses.append("[invalid]")

            return filtered_responses

        filtered_resps = [extract_answer(resp) for resp in resps]

        return filtered_resps


# test cases
if __name__ == "__main__":
    extractor = ExtractAnswerR()
    responses = [
        ["(A) (B)\n (C)"],
        ["a resposta correta é a deltoide."],
        ["a resposta correta é (D) deltoide."],
        ["a resposta é (D) deltoide."],
        ["a resposta correta é (D)."],
        ["a resposta é (D)"],
        ["Nenhuma resposta fornecida"],  # invalid
        ["a resposta é `< > ) ) ] }`"],
        ["Portanto, a resposta correta é (D) 17/06/2019."],
        ["Portanto, a resposta correta é válido"],
        ["A hipótese \"Ser um antepassado de Jonathan é suficiente para ser um amigo próximo de Clifford\" é **inválida**. A resposta correta é **inválido**."],
        ["Vamos pensar passo a passo.\nHora de acordar: 9h.\n9h-12h: comprar uma bicicleta na loja de bicicletas.\n12h-13h: à espera na estação de comboios.\n13h-15h: comprar roupa no centro comercial.\n15h-18h: fazer exercício no ginásio.\n18h-20h: comprar um telemóvel na loja de eletrónica.\n22h: hora de fechar o ateliê de arte.\nA única hora em que a Sarah poderia ter ido ao ateliê de arte é entre as 18h e as 20h, mas como o ateliê de arte fechou às 22h, a resposta correta seria entre as 20h e as 22h. Portanto, a resposta correta é (D) 20h às 22h."],
        ["Vamos analisar a expressão passo a passo:\n\n1. * o resultado é **Verdadeiro**. \n\n\n\n"],
        ["Vamos analisar a expressão passo a passo:\n\n1. * o resultado final é **Verdadeiro**. \n\n\n\n"],
        ["Vamos pensar passo a passo:\n\n(1) Seo de um sabão Lever. O argumento é **válido**. \n\n\n\n"],
        ["Vamos pensar passo a passo.\n\nRecordemos que a ordem das operações em matemática é a seguinte: (1) Parênteses, (2) expoentes, (3) multiplicação e divisão (da esquerda para a direita), (4) adição e subtração (da esquerda para a direita).\n\nEsta equação pode ser escrita como \"A * B\", onde A = (8 - 2 + -2 * 6) e B = (8 + -6 + -8 + -1).\n\nVamos calcular A = (8 - 2 + -2 * 6) = (8 - 2 + (-2 * 6)) = (8 - 2 + (-12)) = (8 - 14) = -6.\n\nVamos calcular B = (8 + -6 + -8 + -1) = (8 + (-6) + (-8) + (-1)) = (2 + (-8) + (-1)) = (-6 + (-1)) = -7.\n\nEntão, a equação final é A * B = -6 * -7 = 42. Portanto, a resposta é 42. \n\n\n\n"],
        ["Vamos pensar passo a passo.\n\n1. **Começamos com 4 pinguins:** Louis, Vincent, Gwen e James.\n\n2. **Eliminamos Bernard:** Isso nos deixa com 3 pinguins.\n\n\n\nPortanto, há **3** pinguins na tabela. A resposta é (C)."],
        ["Vamos pensar passo a passo:\n\n(0) Início:\n* Alice: O Grande Gatsby\n* Bob: Frankenstein\n* Claire: Moby Dick\n* Dave: A Irmandade do Anel\n* Eve: Ardil 22\n\n(1) Alice e Bob trocam:\n* Alice: Frankenstein\n* Bob: O Grande Gatsby\n* Claire: Moby Dick\n* Dave: A Irmandade do Anel\n* Eve: Ardil 22\n\n(2) Dave e Eve trocam:\n* Alice: Frankenstein\n* Bob: O Grande Gatsby\n* Claire: Moby Dick\n* Dave: Ardil 22\n* Eve: A Irmandade do Anel\n\n(3) Eve e Alice trocam:\n* Alice: Ardil 22\n* Bob: O Grande Gatsby\n* Claire: Moby Dick\n* Dave: A Irmandade do Anel\n* Eve: O Grande Gatsby\n\n(4) Dave e Eve trocam:\n* Alice: Ardil 22\n* Bob: O Grande Gatsby\n* Claire: Moby Dick\n* Dave: O Grande Gatsby\n* Eve: A Irmandade do Anel\n\n(5) Claire e Dave trocam:\n* Alice: Ardil 22\n* Bob: O Grande Gatsby\n* Claire: A Irmandade do Anel\n* Dave: Moby Dick\n* Eve: O Grande Gatsby\n\nNo final do semestre, Eve tem **(A) O Grande Gatsby**. \n\n\n\n"],
    ]
    extracted = extractor.apply(responses, [])
    for i, ans in zip(responses, extracted):
        print(f"Response: {i} => Extracted Answer: {ans}")